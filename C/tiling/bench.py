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
		for i in range(1,11):
			n = 2**i
			for j in range(0,16):
				t = 2**j
				if 'multi' in f:
					for k in range(0,cpus):
						thr = 2**k
						os.environ['OMP_NUM_THREADS'] = str(thr)
						for trial in range(5):
							call(["./{}".format(f),str(n),str(t)])
				else:
					for trial in range(5):
						call(["./{}".format(f),str(n),str(t)])
