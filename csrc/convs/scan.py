import math
x = [-0.0384, -1.2903, -2.8832, -0.0598,  0.9360, -0.0032, -0.4952,  0.0296,
        -0.2303,  0.5328,  0.0364, -0.3063,  0.3389,  0.2967, -0.5967,  1.5426]
l = len(x)
w = [1 for i in range(l)]
# x = [i for i in range(l)]
resp = 0
for i in range(len(x)):
	resp += w[i] * x[i]
print('ref:', resp)
print('ref:', sum(x))

resp = 0
aw = [w[j] for j in range(len(x)) if j % 2 == 0]
bw = [w[j] for j in range(len(x)) if j % 2 == 1]
for i in range(int(math.log2(l))):
	ax = [x[j] for j in range(len(x)) if j % 2 == 0]
	bx = [x[j] for j in range(len(x)) if j % 2 == 1]
	x = []
	for j in range(len(bx)):
		if i == 0:
			x.append(bx[j] * bw[j] + ax[j] * aw[j])
		else:
			x.append(bx[j] + ax[j])
print(x)
