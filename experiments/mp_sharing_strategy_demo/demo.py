#!/usr/bin/env python3
"""Demonstrate effectiveness of torch.multiprocessing.set_sharing_strategy('file_system')."""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset


@dataclass
class RunResult:
    strategy: str
    success: bool
    batches_done: int
    batches_target: int
    elapsed_sec: float
    batches_per_sec: float
    peak_fds: Optional[int]
    shm_path: Optional[str]
    shm_base_bytes: Optional[int]
    shm_peak_bytes: Optional[int]
    shm_delta_bytes: Optional[int]
    error: Optional[str]
    fd_limit: Optional[int]
    num_workers: int
    prefetch: int
    tensors_per_sample: int
    batch_size: int
    tensor_shape: str


class MultiTensorDataset(Dataset):
    def __init__(self, length: int, tensors_per_sample: int, tensor_shape: Tuple[int, ...]):
        self.length = length
        self.tensors_per_sample = tensors_per_sample
        self.tensor_shape = tensor_shape

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # Create multiple independent tensors to increase shared-memory segments per batch.
        return tuple(
            torch.empty(self.tensor_shape, dtype=torch.float32)
            for _ in range(self.tensors_per_sample)
        )


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in shape_str.split(",") if p.strip()]
    if not parts:
        return (1024,)
    return tuple(int(p) for p in parts)


def count_open_fds() -> Optional[int]:
    proc_fd = "/proc/self/fd"
    try:
        return len(os.listdir(proc_fd))
    except FileNotFoundError:
        return None


def set_fd_limit(limit: int) -> Optional[int]:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, new_limit))
        return new_limit
    except Exception:
        return None


def get_shm_used_bytes(path: str) -> Optional[int]:
    try:
        stat = os.statvfs(path)
    except Exception:
        return None
    total = stat.f_blocks * stat.f_frsize
    avail = stat.f_bavail * stat.f_frsize
    used = total - avail
    return used


def run_once(args) -> RunResult:
    if args.strategy != "default":
        mp.set_sharing_strategy(args.strategy)

    actual_fd_limit = None
    if args.fd_limit is not None:
        actual_fd_limit = set_fd_limit(args.fd_limit)

    tensor_shape = parse_shape(args.tensor_shape)
    dataset_len = max(args.batches * args.batch_size * 2, args.batch_size * 4)
    dataset = MultiTensorDataset(dataset_len, args.tensors_per_sample, tensor_shape)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    shm_base = None
    shm_peak = None
    shm_path = None
    if args.track_shm:
        shm_path = args.shm_path
        shm_base = get_shm_used_bytes(shm_path)
        shm_peak = shm_base

    peak_fds = count_open_fds()
    batches_done = 0
    t0 = time.perf_counter()
    error = None
    success = True

    try:
        for batch in loader:
            batches_done += 1
            fds_now = count_open_fds()
            if fds_now is not None:
                peak_fds = max(peak_fds or 0, fds_now)
            if args.track_shm:
                shm_now = get_shm_used_bytes(args.shm_path)
                if shm_now is not None:
                    shm_peak = max(shm_peak or 0, shm_now)
            if batches_done >= args.batches:
                break
            # Avoid holding references longer than needed.
            del batch
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - t0

    bps = batches_done / elapsed if elapsed > 0 else 0.0
    shm_delta = None
    if shm_base is not None and shm_peak is not None:
        shm_delta = max(0, shm_peak - shm_base)

    return RunResult(
        strategy=args.strategy,
        success=success,
        batches_done=batches_done,
        batches_target=args.batches,
        elapsed_sec=elapsed,
        batches_per_sec=bps,
        peak_fds=peak_fds,
        shm_path=shm_path,
        shm_base_bytes=shm_base,
        shm_peak_bytes=shm_peak,
        shm_delta_bytes=shm_delta,
        error=error,
        fd_limit=actual_fd_limit,
        num_workers=args.num_workers,
        prefetch=args.prefetch,
        tensors_per_sample=args.tensors_per_sample,
        batch_size=args.batch_size,
        tensor_shape=args.tensor_shape,
    )


def render_table(results: List[RunResult]) -> str:
    headers = [
        "strategy",
        "success",
        "batches",
        "time(s)",
        "batches/s",
        "peak_fds",
        "shm_peak_mb",
        "shm_delta_mb",
        "error",
    ]
    rows = []
    for r in results:
        batches = f"{r.batches_done}/{r.batches_target}"
        shm_peak_mb = (
            "-" if r.shm_peak_bytes is None else f"{r.shm_peak_bytes / (1024 ** 2):.1f}"
        )
        shm_delta_mb = (
            "-"
            if r.shm_delta_bytes is None
            else f"{r.shm_delta_bytes / (1024 ** 2):.1f}"
        )
        rows.append(
            [
                r.strategy,
                "yes" if r.success else "no",
                batches,
                f"{r.elapsed_sec:.2f}",
                f"{r.batches_per_sec:.1f}",
                "-" if r.peak_fds is None else str(r.peak_fds),
                shm_peak_mb,
                shm_delta_mb,
                "-" if r.error is None else r.error,
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(items):
        return "  ".join(item.ljust(col_widths[i]) for i, item in enumerate(items))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in col_widths])]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def run_compare(args) -> int:
    base_args = [
        "--batches",
        str(args.batches),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--prefetch",
        str(args.prefetch),
        "--tensors-per-sample",
        str(args.tensors_per_sample),
        "--tensor-shape",
        args.tensor_shape,
    ]
    if args.fd_limit is not None:
        base_args += ["--fd-limit", str(args.fd_limit)]
    if args.track_shm:
        base_args += ["--shm-path", args.shm_path]
    else:
        base_args.append("--no-track-shm")
    if args.persistent_workers:
        base_args.append("--persistent-workers")
    else:
        base_args.append("--no-persistent-workers")
    if args.pin_memory:
        base_args.append("--pin-memory")

    results = []
    for strategy in ["file_descriptor", "file_system"]:
        cmd = [sys.executable, __file__, "--strategy", strategy, "--json"] + base_args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            print(f"[{strategy}] failed to run: {err}")
            return proc.returncode
        result = json.loads(proc.stdout.strip())
        results.append(RunResult(**result))

    print(render_table(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate how mp.set_sharing_strategy('file_system') can avoid "
            "file-descriptor exhaustion in multi-worker DataLoader."
        )
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "file_descriptor", "file_system"],
        default="default",
        help="Sharing strategy to use for this run.",
    )
    parser.add_argument("--compare", action="store_true", help="Run a comparison.")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--tensors-per-sample", type=int, default=16)
    parser.add_argument("--tensor-shape", type=str, default="1024")
    parser.add_argument("--fd-limit", type=int, default=256)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--shm-path", type=str, default="/dev/shm")

    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=True,
        help="Keep DataLoader workers alive between iterations.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_false",
        dest="persistent_workers",
        help="Disable persistent workers.",
    )
    parser.add_argument(
        "--no-track-shm",
        action="store_false",
        dest="track_shm",
        help="Disable tracking /dev/shm usage.",
    )
    parser.set_defaults(track_shm=True)

    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser


def main() -> int:
    torch.set_num_threads(1)
    args = build_parser().parse_args()

    if args.compare:
        return run_compare(args)

    result = run_once(args)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=True))
        return 0

    print("Run config:")
    print(f"  strategy:          {result.strategy}")
    print(f"  fd_limit:          {result.fd_limit}")
    print(f"  num_workers:       {result.num_workers}")
    print(f"  prefetch:          {result.prefetch}")
    print(f"  tensors/sample:    {result.tensors_per_sample}")
    print(f"  batch_size:        {result.batch_size}")
    print(f"  tensor_shape:      {result.tensor_shape}")
    print(f"  shm_path:          {result.shm_path}")
    print("")
    print("Result:")
    print(render_table([result]))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
