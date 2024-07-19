from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
import random, math, numpy
nn = torch.nn
F = nn.functional
from fused_convs import convs_forward

seed = 1242
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

warpscan_forward = convs_forward

def test_correctness(x, y, atol=1e-1):
	return torch.allclose(x, y, atol=atol)


class Scan(torch.autograd.Function):
	@staticmethod
	def forward(ctx, gates, tokens):
		states = warpscan_forward(gates, tokens)
		# return states.mT.contiguous()
		return states

	@staticmethod
	def backward(ctx, grad_output):
		states, gates = ctx.saved_tensors
		B, C, T = gates.shape

		grad_output = grad_output.contiguous()
		assert states.is_contiguous()
		assert gates.is_contiguous()

		padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
		d_states = scan_forward(padded_shifted_gates, grad_output, reverse=True)

		padded_outputs = torch.cat([torch.zeros_like(states[:, :, :1]), states], dim=-1)[:, :, :-1]
		d_gates = padded_outputs * d_states

		d_tokens = d_states
		return d_gates, d_tokens


def scan2(gates, tokens):
	# on dim 2, kernel_size = 3, stride = 1
	tokens = tokens.unfold(2, 3, 1).transpose(2, 1).contiguous().flatten(2)
	p = 256 - tokens.size(-1)
	tokens = F.pad(tokens, (0, p))
	B, T, C = tokens.shape
	gates = F.pad(gates.flatten(1), (0, p))
	# print(tokens.shape)
	# print(gates.shape)
	# exit()
	# res = (tokens.view(tokens.size(0), tokens.size(1), -1, 32) * gates.view(1, 1, -1, 32))
	# gates = torch.randn(64, 256).to('cuda')
	# tokens = torch.randn(1, 2, 256).to('cuda')
	res = (gates.view(1, 1, 1, -1, 256) * tokens.view(B, T, 1, 256)).sum(-1).view(B, T, gates.size(0))
	res2 = Scan.apply(gates, tokens).mT.contiguous()
	# print(res[0,0])
	# print(res2[0,0])
	# print(res.shape)
	# print(res2.shape)
	# print(test_correctness(res.mT.contiguous(), res2, atol=1e-3))
	# exit()
	return res2


fused_conv1d = Scan.apply
def prepare_data(gates, tokens):
	tokens = tokens.unfold(2, 3, 1).transpose(2, 1).contiguous().flatten(2)
	p = 256 - tokens.size(-1)
	gates = F.pad(gates.flatten(1), (0, p))
	tokens = F.pad(tokens, (0, p))
	return gates, tokens


def fused_conv(gates, tokens):
	gates, tokens = prepare_data(gates, tokens)
	return Scan.apply(gates, tokens)


def ref_conv(x, conv1, chunklen=2):
	return F.conv1d(x, conv1.weight.data)

if __name__ == "__main__":
	B, T, C = 1, 80, 3000
	c_out = 64
	conv1 = nn.Conv1d(T, c_out, kernel_size=3, bias=False).to('cuda')
	for _ in range(3):
		x = torch.randn(B, T, C).to('cuda')
		ref = ref_conv(x, conv1)
		fused = fused_conv(conv1.weight.data, x).mT.contiguous() # + conv1.bias.data.view(1, c_out, 1)
		print(test_correctness(ref, fused, atol=1e-3))
		print(test_correctness(ref, fused))

