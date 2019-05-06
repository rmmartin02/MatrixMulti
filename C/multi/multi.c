#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <x86intrin.h>

//Intel® Core™ i7-4720HQ CPU
//

//#define BASE
//#define V1 
#define V2
//#define V3
//#define V4
//#define V5
//#define V6
//#define V7
 
#define N 1
const char *usage = "%s <square matrix one dimension size>";

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp; 
  int stat;
  double t;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return t = (Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void Transpose(int size, float* m)
{
  int i, j;
  for (i = 0; i < size; i++) {
    for (j = i + 1; j < size; j++) {
      float temp;
      temp = m[i*size+j];
      m[i*size+j] = m[j*size+i];
      m[j*size+i] = temp;
    }
  }
}

#ifdef BASE

/*Baseline Matrix Multiplication*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      C[i*size+j] = 0;
      for (k = 0; k < size; k++) {
        C[i*size+j] += A[i*size+k] * B[k*size+j];
      }
    }
  }
}


#elif defined V1

/*Opt1: multithreaded using openmp*/
void mm(int size, float* A, float* B, float* C){
int tid, nthreads, i, j, k;
int const NRA = size;
int const NCA = size;
int const NCB = size;
#pragma omp parallel shared(A,B,C,nthreads) private(tid,i,j,k)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Starting matrix multiple example with %d threads\n",nthreads);
    //printf("Initializing matrices...\n");
    }

  /*** Do matrix multiply sharing iterations on outer loop ***/
  /*** Display who does which iterations for demonstration purposes ***/
  //printf("Thread %d starting matrix multiply...\n",tid);
  #pragma omp for schedule (auto)
  for (i=0; i<NRA; i++)
    {
    //printf("Thread=%d did row=%d\n",tid,i);
    for(j=0; j<NCB; j++)
      for (k=0; k<NCA; k++)
        C[i*size+j] += A[i*size+k] * B[k*size+j];
    }
  }
}

#elif defined V2

/*Opt1: multithreaded using openmp and vectorized*/
void mm(int size, float* A, float* B, float* C){
int tid, nthreads, i, j, k;
__m256 a_vec, b_vec, c_vec, temp_vec;
int const NRA = size;
int const NCA = size;
int const NCB = size;
#pragma omp parallel shared(A,B,C,nthreads) private(tid,i,j,k, a_vec, b_vec, c_vec, temp_vec)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Starting matrix multiple example with %d threads\n",nthreads);
    //printf("Initializing matrices...\n");
    }

  /*** Do matrix multiply sharing iterations on outer loop ***/
  /*** Display who does which iterations for demonstration purposes ***/
  //printf("Thread %d starting matrix multiply...\n",tid);
  #pragma omp for schedule (auto)
  for (i=0; i<NRA; i++)
    {
    for(j=0; j<NCB; j+=8){
      c_vec = _mm256_setzero_ps();
      for (k=0; k<NCA; k++){
        a_vec = _mm256_loadu_ps(&A[i*size+k]);
        b_vec = _mm256_loadu_ps(&B[k*size+j]);
        //multiply then add
        temp_vec = _mm256_mul_ps(a_vec,b_vec);
        c_vec = _mm256_add_ps(c_vec, temp_vec);
      }
      _mm256_storeu_ps(&C[i*size+j], c_vec);
    }
    }
  }
}

#endif

int main(int argc, char *argv[]) {
  int msize;
  __attribute__((aligned(16)))  float *A, *B, *C;
  double begin, last;
  double cputime, gflops;
  int i;

  if (argc!=3) {
    printf(usage, argv[0]);
    return 0;
  }

  msize = atoi(argv[1]);
  omp_set_num_threads(atoi(argv[2]));

  assert(msize>0);
  
  A = (float *) _mm_malloc (msize * msize * sizeof(float), 16);
  B = (float *) _mm_malloc (msize * msize * sizeof(float), 16);
  C = (float *) _mm_malloc (msize * msize * sizeof(float), 16);

  assert(A && B && C);

  memset(A, 0, msize * msize * sizeof(float));
  memset(B, 0, msize * msize * sizeof(float));
  memset(C, 0, msize * msize * sizeof(float));

  int ii,jj;
  for(ii = 0; ii < msize; ii++){
    for (jj = 0; jj < msize; jj++){
      A[ii*msize + jj] = (ii + jj) * 0.1;
    }
  }

  for(ii = 0; ii < msize; ii++){
    for (jj = 0; jj < msize; jj++){
      B[ii*msize + jj] = (ii - jj) * 0.2;
    }
  }
  
  float rets = 0.0;
  begin = rtclock();
  
  for(i=0; i<N; i++) {
    mm(msize, A, B, C);
    rets += C[i];
  }

  last = rtclock();
  cputime = (last - begin) / N;

  gflops = (2.0 * msize * msize * msize / cputime) / 1000000000.0;
  printf("Time=%lfms GFLOPS=%.3lf\n", cputime*1000, gflops);
  printf("Prove optimization out: rets = %lf\n", rets);
  
  FILE *fp;
  fp = fopen("test.txt", "a");
  fprintf(fp, "%s, %s, %d, %lf, %.3lf, %lf\n", argv[0], argv[1], omp_get_max_threads() , cputime*1000, gflops, rets);
  fclose(fp);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  return 0;  
}