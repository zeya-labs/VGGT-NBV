import sys

import torch

from nbv_framework.training.loss.Density_aware_Chamfer_Distance.utils_v2.model_utils import calc_emd


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is required to run the EMD kernel.", file=sys.stderr)
        return

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Make only a single point nearly identical between pred/gt and sweep deltas.
    # sqrt(dist) has an infinite gradient at 0; very small deltas can underflow
    # to 0 in float32 and trigger NaNs inside the CUDA kernel.
    batch_size = 1
    num_points = 1024  # keep it a multiple of 1024 for the EMD kernel
    base_pred = torch.rand(batch_size, num_points, 3, device=device) * 0.5 + 0.5
    base_gt = torch.rand(batch_size, num_points, 3, device=device) * 0.5 + 0.5
    shared_point = torch.tensor([0.7, 0.8, 0.9], device=device)
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

    any_nan = False
    for delta in deltas:
        pred = base_pred.clone()
        gt = base_gt.clone()

        pred[0, 0] = shared_point
        gt[0, 0] = shared_point + torch.tensor([delta, 0.0, 0.0], device=device)
        pred.requires_grad_()

        status = "ok"
        loss_val = float("nan")
        has_nan = False
        has_inf = False
        try:
            loss = calc_emd(pred, gt, eps=0.005, iterations=50).mean()
            loss_val = loss.item()
            loss.backward()
            torch.cuda.synchronize()
            grad = pred.grad
            if grad is None:
                status = "no_grad"
                has_nan = True
            else:
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
        except RuntimeError as exc:
            status = f"backward_error:{type(exc).__name__}"
            has_nan = True
            has_inf = True

        effective_delta = (gt[0, 0] - pred[0, 0]).abs().max().item()

        any_nan = any_nan or has_nan
        print(
            f"delta={delta:.1e} eff_delta={effective_delta:.1e} "
            f"loss={loss_val:.6f} nan={has_nan} inf={has_inf} status={status}"
        )

    if not any_nan:
        print(
            "NOTE: If no NaN appears, try smaller deltas or increase iterations.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
