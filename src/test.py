import torch

a = torch.randn([2, 2])
b = torch.randn([2, 2])

print(a)
print(b)
c = torch.min(a, b)
print(c)