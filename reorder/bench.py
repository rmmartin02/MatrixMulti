import os
from subprocess import call

with open('test.txt','w') as f:
	f.write('')

for f in os.listdir():
	if '.' not in f:
		for i in range(1,11):
			n = 2**i
			for trial in range(5):
				call(["./{}".format(f),str(n)])
