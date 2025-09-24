import torch

# 初始设置
a = torch.zeros((), requires_grad=True)
b = torch.ones((), requires_grad=True)

print(f"Before op: a.requires_grad={a.requires_grad}, a.is_leaf={a.is_leaf}, a.grad_fn={a.grad_fn}")
# 输出: Before op: a.requires_grad=False, a.is_leaf=True, a.grad_fn=None

# 执行原地操作
a += b

print(f"After op: a.requires_grad={a.requires_grad}, a.is_leaf={a.is_leaf}, a.grad_fn={a.grad_fn}")
# 输出: After op: a.requires_grad=True, a.is_leaf=False, a.grad_fn=<AddBackward0 object at ...>

# 现在调用 backward() 不会报错
print("\nCalling a.backward()...")
a.backward()
print("a.backward() executed successfully.")

# 检查梯度
print(f"a.grad: {a.grad}")  # a 不是叶子节点，梯度不会累积在.grad中
# 输出: a.grad: None

print(f"b.grad: {b.grad}")  # b 是需要梯度的叶子节点，梯度会累积
# 输出: b.grad: tensor(1.)