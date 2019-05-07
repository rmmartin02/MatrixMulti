import os
from subprocess import call
from multiprocessing import cpu_count
from math import log2

cpus = int(log2(cpu_count()))+1
print(cpus)

with open('test.txt','w') as f:
	f.write('')

for f in os.listdir():
	if 'sgemm' in f:
		for i in range(1,14):
			n = 2**i
			for trial in range(5):
				call(["./{}".format(f),str(n)])