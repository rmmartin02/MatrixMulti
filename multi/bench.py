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
			for j in range(0,cpus):
				t = 2**j
				call(["./{}".format(f),str(n),str(t)])
