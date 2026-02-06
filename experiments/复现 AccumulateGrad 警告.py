import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

torch.manual_seed(0)
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(True)

device = "cuda"
w = torch.nn.Parameter(torch.randn(1024, 1024, device=device))
opt = torch.optim.SGD([w], lr=1e-3)

s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
keep = []

def step(stream):
    opt.zero_grad(set_to_none=True)
    with torch.cuda.stream(stream):
        loss = (w @ w).sum()
    torch.cuda.current_stream().wait_stream(stream)

    # 保留图，确保 AccumulateGrad 节点跨迭代存活
    keep.append(loss)
    loss.backward(retain_graph=True)
    opt.step()

step(s0)
step(s1)

print("done")
