# mp.set_sharing_strategy('file_system') 直观量化演示

这个小项目通过 **限制进程可用文件描述符数量** + **制造大量共享内存段**，直观展示：
- 默认共享策略（`file_descriptor`）更容易触发 `Too many open files`
- 设置 `mp.set_sharing_strategy('file_system')` 后可以显著降低失败率

## 核心思路（为什么直观）
- DataLoader 多进程会把 batch 中的 tensor 放进共享内存。
- 默认策略 `file_descriptor` 需要为大量共享内存段维持文件描述符。
- 当 `fd` 限制很小且 batch 中“tensor 数量很多”时，**会更容易耗尽 fd**。
- `file_system` 策略用文件系统路径来共享内存，能显著缓解 fd 枯竭问题。

## 运行方式
推荐直接对比：

```bash
python experiments/mp_sharing_strategy_demo/demo.py --compare \
  --fd-limit 256 \
  --num-workers 8 \
  --prefetch 8 \
  --tensors-per-sample 16 \
  --tensor-shape 1024 \
  --batch-size 64 \
  --batches 200
```

输出会是一个对照表：
- `batches`：实际完成批次数 / 目标批次数
- `peak_fds`：主进程观测到的峰值 fd 数量
- `shm_peak_mb`：观测到的 `/dev/shm` 峰值占用（MB）
- `shm_delta_mb`：相对运行前基线的增量（MB）
- `error`：如果失败，显示错误（通常是 `Too many open files`）

## 自定义参数
- `--fd-limit`：人为降低 fd 上限，确保现象更稳定
- `--tensors-per-sample`：每个样本返回多少个 tensor
- `--tensor-shape`：单个 tensor 大小（例如 `1024` 或 `3,256,256`）
- `--num-workers` / `--prefetch`：增加并发和预取，会放大差异
- `--shm-path`：共享内存挂载点（默认 `/dev/shm`）
- `--no-track-shm`：关闭 `/dev/shm` 统计

## 单次运行（不比较）
```bash
python experiments/mp_sharing_strategy_demo/demo.py --strategy file_system
```

## 预期现象（可量化）
- `file_descriptor`：在低 fd 限制下，更容易提前失败，`batches` 低，`error` 非空
- `file_system`：更容易完整跑完 `batches`，`error` 为空

## 注意（环境限制）
如果看到 `PermissionError: [Errno 13] Permission denied`，通常是系统禁用了
POSIX 共享内存或信号量（例如受限容器环境）。此时任何多进程 DataLoader
都会失败，建议在常规 Linux 环境或放开 `/dev/shm` / 信号量限制后再运行。

如果你需要更强对比，可以进一步提高 `--tensors-per-sample` 或 `--prefetch`。
