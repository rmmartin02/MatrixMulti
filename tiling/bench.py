import os
from subprocess import call
from multiprocessing import cpu_count
from math import log2

cpus = int(log2(cpu_count()))+1
print(cpus)

with open('test.txt','w') as f:
	f.write('')

for f in os.listdir():
	if '.' not in f:
		for i in range(1,12):
			n = 2**i
			for j in range(0,16):
				t = 2**j
				if 'multi' in f:
					for k in range(0,cpus):
						thr = 2**k
						os.environ['OMP_NUM_THREADS'] = str(thr)
						call(["./{}".format(f),str(n),str(t)])
				else:
					call(["./{}".format(f),str(n),str(t)])
