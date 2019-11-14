#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define EPS (0.0000001f)

#define M (4*1048576)
__device__ int gl[M];


__device__ float Cp[M]; 


// Thomas resolution for tridiagonal symmetric matrices
__global__ void thom_sym_k(float *S, float *D, float *Y, int n){

	// The global memory access index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j;


	int first = idx*n;
	int first_c = idx*n + 1;

	float d = D[first];
	Cp[first_c] = S[first_c] / d; 
	Y[first] = Y[first] / d;

	for (j=1;j<n-1;j++) { //Decomposition and forward substitution.
		Cp[first_c + j] = S[first_c + j] / (D[first + j] - S[first_c + j - 1]*Cp[first_c + j - 1]);	
		Y[first + j] = (Y[first + j] - S[first_c + j - 1]*Y[first + j - 1]) / (D[first + j] - S[first_c + j - 1]*Cp[first_c + j - 1]);
	}

	// One more iteration for Y (j=n-1)
	Y[first + n - 1] = (Y[first + n - 1] - S[first_c + n - 2]*Y[first + n - 2]) / (D[first + n - 1] - S[first_c + n - 2]*Cp[first_c + n - 2]);

	for (j=(n-2);j>=0;j--){ //Backsubstitution.
		Y[first + j] = Y[first + j] - Cp[first_c + j]*Y[first + j + 1];
	}
}

// Parallel cyclic reduction for tridiagonal symmetric matrices
__global__ void pcr_sym_k(float *a, float *b, float *y, int n)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n;

	// Identifies the data concerned by the computations
	int Qt = (threadIdx.x - tidx) / n;

	// The global memory access index
	int gb_index_x = Qt + blockIdx.x*(blockDim.x / n);

	int i; 
	int nt = 4*Qt*n;
	int d = (n / 2 + (n % 2))*(tidx % 2) + (int)tidx / 2;
	
	// Shared memory
	extern __shared__ float sAds[];

	float *sa = (float*)&sAds[nt];
	float *sb = (float*)&sa[n];
	float *sy = (float*)&sb[n];
	int *sl = (int*)&sy[n];

	sa[tidx] = a[gb_index_x*n + tidx];
	sb[tidx] = b[gb_index_x*n + tidx];
	sy[tidx] = y[gb_index_x*n + tidx];
	sl[tidx] = tidx;

	int lL, aL, bL, yL, bLp, tl, tr; // local variables

	//////////////////////////////////////////////////////////////
	//
	//	Step 2:	Fill with your code : Additional variables definition 
	//						  and copy the values in shared 
	//
	//////////////////////////////////////////////////////////////
	__syncthreads();

	tl = tidx-1;
	tr = tidx+1;
	if(tl<0) tl=0;
	if(tr>=n) tr=0;

	for (i = 0; i < (int)log2((float)n) + 1; i++){
		lL = sl[tidx];
		aL = sa[tidx];
		bL = sb[tidx];
		yL = sy[tidx];
		bLp = sb[tl];


		//Reduction phase
		//////////////////////////////////////////////////////////////
		//
		//		Step 2:	Fill with your code
		//
		//////////////////////////////////////////////////////////////
		__syncthreads();
		if (i < (int)log2((float)n)){//Permutation phase
			sa[d] = aL;
			sa[d] = bL;
			sa[d] = yL;
			sa[d] = lL;
			__syncthreads();
		}
	}

	//////////////////////////////////////////////////////////////
	// écrire la solution de shared vers la mémoire globale 
	//////////////////////////////////////////////////////////////
}


// Produces tridiagonal symmetric diagonally dominant matrices 
__global__ void Tri_k(float *D, float *S, float norm, int i, 
						   int n, int L)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n;
	// Identifies the data concerned by the computations
	int Qt = (threadIdx.x - tidx) / n;
	// The global memory access index
	int gb_index_x = Qt + blockIdx.x*(blockDim.x / n);

	if(gb_index_x*n + tidx + i < L){
		D[gb_index_x*n + tidx + i] = ((float)tidx+1.0f)/(norm);
		if (tidx > 0){
			S[gb_index_x*n + tidx + i] = ((float)tidx+1.0f)/(norm*3);
		}else{S[gb_index_x*n + tidx + i] = 0.0f;}
	}
}


int main(){

	int i, j;

	// The rank of the matrix
	int Dim = 64;
	
	// The number of blocks
	int NB = M/Dim;
	
	// The number of matrices to invert
	int size = NB;

	// The diagonal elements
	float *D, *DGPU;
	// The subdiagonal elements
	float *S, *SGPU;
	// The system vector
	float *Y, *YGPU;

	float TimerV;					// GPU timer instructions
	cudaEvent_t start, stop;		// GPU timer instructions
	cudaEventCreate(&start);		// GPU timer instructions
	cudaEventCreate(&stop);			// GPU timer instructions

	// Memory allocation
	D = (float *)calloc(size*Dim,sizeof(float));
	S = (float *)calloc(size*Dim,sizeof(float));
	Y = (float *)calloc(size*Dim,sizeof(float));
	cudaMalloc(&DGPU, size*Dim*sizeof(float));
	cudaMalloc(&SGPU, size*Dim*sizeof(float));
	cudaMalloc(&YGPU, size*Dim*sizeof(float));

	// Tridiagonal elements
	int HM = M/(NB*Dim);
	for (i=0; i*Dim*NB<M; i++){
		Tri_k <<<NB,HM*Dim>>>(DGPU, SGPU, 10.0f, i*Dim*NB, Dim, 
										  Dim*NB);
	}

	cudaMemcpy(D, DGPU, size*Dim*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(S, SGPU, size*Dim*sizeof(float), cudaMemcpyDeviceToHost);

	// Second member
	for (i=0; i<size; i++){
		for (j=0; j<Dim; j++){
			Y[j+i*Dim]=0.5f*j;
		}
	}
	cudaMemcpy(YGPU,Y,size*Dim*sizeof(float),cudaMemcpyHostToDevice);


	// Resolution part
	cudaEventRecord(start,0);


	/////////////////////////////////////////////////////////////////////
	// Step 2:	PCR
	/////////////////////////////////////////////////////////////////////
	// The minimum group of threads per block for PCR /!\ Has to be chosen by students
	// /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\

	/*
	int minTB = 1;  // Choose the right value with respect to Dim
	printf("%i \n", minTB);
	pcr_sym_k<<<NB/minTB,Dim*minTB,4*minTB*Dim*sizeof(float)>>>(SGPU, DGPU, YGPU, Dim);
	*/

	/////////////////////////////////////////////////////////////////////
	// Step 1:	Thomas
	/////////////////////////////////////////////////////////////////////
	thom_sym_k<<<NB/256,256>>>(SGPU, DGPU, YGPU, Dim);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&TimerV,start, stop);

	cudaMemcpy(Y, YGPU, size*Dim*sizeof(float), cudaMemcpyDeviceToHost);
	for (i=0; i<size; i++){
	        if(i==573){
			printf("\n\n");
			for (j=0; j<Dim; j++){
				printf("%.5e, ",Y[j+i*Dim]);
			}
		}
	}


	printf("Execution time: %f ms\n", TimerV);

	// Memory free for other arrays
	free(D);
	cudaFree(DGPU);
	free(S);
	cudaFree(SGPU);
	free(Y);
	cudaFree(YGPU);

	cudaEventDestroy(start);		// GPU timer instructions
	cudaEventDestroy(stop);			// GPU timer instructions

	return 0;
}
