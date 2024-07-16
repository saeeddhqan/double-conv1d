import torch

B, T, C = 2, 8, 30
C_out = 1

a = torch.randn(B, T, C)
ac = a.unfold(2, 3, 1).flatten(2)
c = torch.nn.Conv1d(T, C_out, kernel_size=3)
w = c.weight.data
bias = c.bias.data
print(bias.shape)
ref = c(a)
out = torch.zeros(B, T, ac.size(-1))
print(out.shape)
for co in range(C_out):
	for ci in range(ac.size(-1)):
		print(co, ci)
		out[:,co,ci] = bias[co] + w[co, :, ci % 3] * ac[:, :, ci]


out = out.view(B, T, out.size(-1), 3).sum(dim=-1)

print(out.shape)
assert torch.allclose(ref, out, atol=1e-1)
