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
//#define V2
//#define V3
//#define V4
//#define V5
#define V6
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

/*Opt1: transpose B to improve data locality*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k;
  Transpose(size, B);
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      float c = 0;
      for (k = 0; k < size; k++) {
        c += A[i*size+k] * B[j*size+k];
      }
      C[i*size+j] = c;
    }
  }
  Transpose(size, B);
}

#elif defined V2

/*Opt2: transpose B and tile to improve data locality*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k, i1, j1, k1;
  int t = 1024;
  Transpose(size, B);
  for (i = 0; i < size; i+=t) {
    for (j = 0; j < size; j+=t) {
      for (k = 0; k < size; k+=t) {
        for (i1 = i; i1 < i+t; i1++){
          for (j1= j; j1 < j+t; j1++){
            float c = C[i1*size+j1];
            for (k1 = k; k1 < k+t; k1++) {
              c += A[i1*size+k1] * B[j1*size+k1];
            }
            C[i1*size+j1] = c;
          }
        }
      }
    }
  }
  Transpose(size,B);
}

#elif defined V3

/*Opt3: transpose B and vectorize*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k;
  __m128 c_vec, a_vec, b_vec, temp_vec;
  Transpose(size, B);
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j+=4) {
      c_vec = _mm_setzero_ps();
      for (k = 0; k < size; k++) {
        a_vec = _mm_loadu_ps(&A[i*size+k]);
        b_vec = _mm_loadu_ps(&B[j*size+k]);
        //multiply then add
        temp_vec = _mm_mul_ps(a_vec,b_vec);
        c_vec = _mm_add_ps(c_vec, temp_vec);
      }
      //printf("%f %d\n",C[i*size+j],i*size+j);
      //C[i*size+j] = 1;
      _mm_storeu_ps(&C[i*size+j], c_vec);
    }
  }
  Transpose(size, B);
}

#elif defined V4

/*Opt3: transpose B and vectorize*/
void mm(int size, float* A, float* B, float* C)
{
  int i, j, k;
  __m256 c_vec, a_vec, b_vec, temp_vec;
  Transpose(size, B);
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j+=8) {
      c_vec = _mm256_setzero_ps();
      for (k = 0; k < size; k++) {
        a_vec = _mm256_loadu_ps(&A[i*size+k]);
        b_vec = _mm256_loadu_ps(&B[j*size+k]);
        //multiply then add
        temp_vec = _mm256_mul_ps(a_vec,b_vec);
        c_vec = _mm256_add_ps(c_vec, temp_vec);
      }
      //printf("%f %d\n",C[i*size+j],i*size+j);
      //C[i*size+j] = 1;
      _mm256_storeu_ps(&C[i*size+j], c_vec);
    }
  }
  Transpose(size, B);
}

#elif defined V5

void mm(int size, float* A, float* B, float* C){
int	tid, nthreads, i, j, k;
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

#elif defined V6

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

  if (argc!=2) {
    printf(usage, argv[0]);
    return 0;
  }

  msize = atoi(argv[1]);

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

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  return 0;  
}