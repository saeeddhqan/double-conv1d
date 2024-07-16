
from pathlib import Path

import torch
import math
from fused_convs import convs_forward
from einops import rearrange
import random, math, numpy
nn = torch.nn
F = nn.functional

seed = 1244
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def test_correctness(x, y, atol=1e-1):
	assert torch.allclose(x, y, atol=atol), 'Tensor mismatch'



class FusedConv(torch.autograd.Function):
	idxs = None
	@staticmethod
	def forward(ctx, x, w, b):
		x = x.unfold(2, 3, 1).flatten(2).mT.contiguous().view(x.size(0), x.size(-1)-2, x.size(-2) * 3)
		r = convs_forward(x, w, b)
		# ctx.save_for_backward(x, w, b)
		return r

	@staticmethod
	def backward(ctx, dout):
		# x, w, b = ctx.saved_tensors
		# dout  = dout.contiguous()
		# du, dk, dbias = dimwise_backward(dout, input, weight, bias, ctx.padding, ctx.is_bhl)
		return None, None, None


fused_conv = FusedConv.apply

def conv_ref(x, conv1, chunklen=2):
	return F.conv1d(x, conv1.weight.data, conv1.bias.data)

if __name__ == "__main__":
	B, T, C = 2, 4, 10
	conv1 = nn.Conv1d(T, 2, kernel_size=3).to('cuda')
	for _ in range(3):
		x = torch.randn(B, T, C).to('cuda')
		ref = conv_ref(x, conv1)
		print(conv1.weight.data.shape)
		fused = fused_conv(x, conv1.weight.data, conv1.bias.data)
		print(ref.shape)
		print(fused.shape)
		print(ref[1][0])
		print(fused[1][0])
		test_correctness(ref, fused, atol=1e-3)
		test_correctness(ref, fused)
		print('passed: similar output')
