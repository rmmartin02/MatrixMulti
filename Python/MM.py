import numpy as np
import time
import threading
from multiprocessing import cpu_count
from math import log2

cpus = int(log2(cpu_count()))+1
print(cpus)


def checkMM(O1, O2, e):
    O1 = np.array(O1)
    O2 = np.array(O2)
    for i in range(len(O1)):
        for j in range(len(O1[i])):
            if abs(O1[i][j]-O2[i][j])>e:
                return False
    return True


def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))] 


def multi(core, a, b, c):
    n = len(A)
    # Each thread computes 1/4th of matrix multiplication 
    for idx in range(core*n//4, (core+1)*n//4):
        for jdx in range(n):
            for kdx in range(n):
                c[idx][jdx] += a[idx][kdx] * b[kdx][jdx]


with open('test.txt','w') as f:
    f.write('')

for abc in range(1,12):
    print(2**abc)
    for trial in range(3):
        N = 2**abc
        A = np.random.rand(N, N).astype(np.dtype('f4'))
        B = np.random.rand(N, N).astype(np.dtype('f4'))
        C = np.zeros((N, N))
        AL = A.tolist()
        BL = B.tolist()
        CL = C.tolist()
        correct = A.dot(B)

        # default implementation
        print('default')
        start = time.clock()
        for i in range(N):
            for j in range(N):
                CL[i][j] = 0.0
                for k in range(N):
                    CL[i][j] += AL[i][k] * BL[k][j]

        cputime = time.clock()-start
        with open('test.txt','a') as f:
            f.write('{}, {}, {}, {}\n'.format('default', N, cputime ,np.array(CL).sum()))
        CL = C.tolist()

        # transposed default
        print('transposed')
        start = time.clock()
        BL = transpose(BL)
        for i in range(N):
            for j in range(N):
                c = 0.0
                for k in range(N):
                    c += AL[i][k] * BL[j][k]
                CL[i][j] = c
        BL = transpose(BL)
        cputime = time.clock()-start
        with open('test.txt','a') as f:
            f.write('{}, {}, {}, {}\n'.format('transposed', N, cputime ,np.array(CL).sum()))
        CL = C.tolist()

        # Tiled
        print('tiled')
        for tilesize in range(11):
            start = time.clock()
            t = 2**tilesize
            for i in range(0, N, t):
                for j in range(0, N, t):
                    for k in range(0, N, t):
                        for i1 in range(i, min(i+t, N)):
                            for j1 in range(j, min(j+t, N)):
                                c = CL[i1][j1]
                                for k1 in range(k, min(k+t, N)):
                                    c += AL[i1][k1] * BL[k1][j1]
                                CL[i1][j1] = c
            cputime = time.clock()-start
            with open('test.txt','a') as f:
                f.write('{}, {}, {}, {}, {}\n'.format('tiled', N, t,cputime ,np.array(CL).sum()))
            CL = C.tolist()

        # Multi-threaded
        # https://www.geeksforgeeks.org/multiplication-matrix-using-pthreads/
        print('bad multithread')
        start = time.clock()
        # declaring four threads
        MAX_THREAD = 4

        # Creating four threads, each evaluating its own part 
        threads = []
        for i in range(MAX_THREAD):
            t = threading.Thread(target=multi, args=(i, AL, BL, CL))
            t.start()
            threads.append(t)

        # joining and waiting for all threads to complete 
        for x in threads:
            x.join()

        cputime = time.clock()-start
        with open('test.txt','a') as f:
            f.write('{}, {}, {}, {}\n'.format('4_thread', N, cputime ,np.array(CL).sum()))
        CL = C.tolist()

        # different mulithreading
        # https://github.com/mtrebi/matrix-multiplication-threading
        print('better multithread')
        def multiply_threading(result, thread_number, m1, m2):
            # Calculate workload
            MATRIX_SIZE = len(m1)
            n_elements = (MATRIX_SIZE * MATRIX_SIZE)
            n_operations = n_elements / THREADS_NUMBER
            rest_operations = n_elements % THREADS_NUMBER
            
            if thread_number == 0:
                # First thread does more job
                start_op = n_operations * thread_number
                end_op = (n_operations * (thread_number + 1)) + rest_operations
            else:
                start_op = n_operations * thread_number + rest_operations
                end_op = (n_operations * (thread_number + 1)) + rest_operations

            for op in range(int(start_op), int(end_op)):
                row = op % MATRIX_SIZE
                col = op // MATRIX_SIZE
                r = 0.0
                for idx in range(MATRIX_SIZE):
                    e1 = m1[row][idx]
                    e2 = m2[idx][col]
                    r += e1 * e2
                result[row][col] = r

        start = time.clock()
        for tn in range(cpus):
            THREADS_NUMBER = 2**tn
            print('Threads',THREADS_NUMBER)
            threads = []
            for numThread in range(THREADS_NUMBER):
                t = threading.Thread(target=multiply_threading, args=(CL, numThread, AL, BL))
                t.start()
                threads.append(t)

            # joining and waiting for all threads to complete
            for x in threads:
                x.join()
                
            cputime = time.clock()-start
            with open('test.txt','a') as f:
                f.write('{}, {}, {}, {}, {}\n'.format('multithread', N, THREADS_NUMBER,cputime ,np.array(CL).sum()))
            CL = C.tolist()