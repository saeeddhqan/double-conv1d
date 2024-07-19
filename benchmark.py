
import torch
nn = torch.nn
F = nn.functional

from eval import warp
# from csrc.convs import warp
fused_conv1d = warp.fused_conv1d
prepare_data = warp.prepare_data
import random, math, numpy, time, sys
import matplotlib.pyplot as plt
from prettytable import PrettyTable


seed = 1244
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def plot_vs(x, y1_mean, y2_mean):
	plt.plot(x, y1_mean, linestyle='-', label='fused conv1d')
	plt.plot(x, y2_mean, linestyle='-', label='conv1d')
	plt.legend()
	plt.title('conv1d versus fused conv1d')
	plt.xlabel('seqlen')
	plt.ylabel('time')
	plt.savefig(f'vg_seqlen.png')
	plt.clf()


if __name__ == "__main__":
	B, T, D = 1, 256, 2
	d_in = 1
	steps = 100
	vanilla = False

	# filter_lens = [8, 64, 128, 256, 512, 768]
	filter_lens = [64, 128, 256, 512]
	dlen =  [100, 200, 300, 500]

	results = PrettyTable()
	results.field_names = ['B', 'F', 'D', 'conv1d (ms)', 'fused conv1d (ms)', 'speedup']

	lmean1 = []
	lmean2 = []
	speedups1 = []
	for c, c_out in enumerate(filter_lens):
		conv = nn.Conv1d(80, c_out, 3, bias=False).to('cuda')
		tl_mean1 = torch.empty(len(dlen))
		tl_mean2 = torch.empty(len(dlen))
		for d, D in enumerate(dlen):
			timing1 = torch.empty(10)
			timing2 = torch.empty(10)

			for r in range(timing1.size(0)):
				x = torch.randn(B, 80, D, requires_grad=False).to('cuda')



				with torch.no_grad():
					w, y = prepare_data(conv.weight.data, x)
					w2 = conv.weight.data
					y2 = x
					# warmup
					fused_conv1d(w, y)
					F.conv1d(y2, w2)

					start_time = time.perf_counter()
					for _ in range(steps):
						fused_conv1d(w, y)
					timing1[r] = time.perf_counter() - start_time

					start_time = time.perf_counter()
					for _ in range(steps):
						F.conv1d(x, w2)
					timing2[r] = time.perf_counter() - start_time


			t1 = timing1.mean().item() # fused
			t2 = timing2.mean().item() # simple
			speedup1 = t2 / t1
			results.add_row([B, c_out, D, t2, t1, speedup1])
			print(f"B={B}, F={c_out}, D={D}, speedup={speedup1}, fused={t1}, simple={t2}")

			speedups1.append(speedup1)
			tl_mean1[d] = t1
			tl_mean2[d] = t2
		lmean1.append(tl_mean1.mean().item())
		lmean2.append(tl_mean2.mean().item())

	results.float_format = '0.4'
	print(results)
	results = PrettyTable()
	results.field_names = ['conv1d (ms)', 'fused conv1d (ms)', 'speedup']

	overall_t1 = sum(lmean1) / len(lmean1)
	overall_t2 = sum(lmean2) / len(lmean2)
	s1 = sum(speedups1) / len(speedups1)
	results.add_row([overall_t2, overall_t1, s1])
	
	results.float_format = '0.4'
	print(results)

	plot_vs(filter_lens, lmean1, lmean2)
