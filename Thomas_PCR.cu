#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define EPS (0.0000001f)
#define M (4*1048576)

__device__ float Cp[M];

// Produces tridiagonal symmetric diagonally dominant matrices
__global__ void Tri_k(float *sub, float *diag, float norm, int i, int n, int total_length)
{
	// Identifies the thread working within a group
	int thread_grp_idx = threadIdx.x%n;
	// Identifies the data concerned by the computations
	int data_range = (threadIdx.x - thread_grp_idx) / n;
	// The global memory access index
	int gb_index_x = data_range + blockIdx.x*(blockDim.x / n);

	if (gb_index_x*n + thread_grp_idx + i < total_length) {
		diag[gb_index_x*n + thread_grp_idx + i] = ((float)thread_grp_idx + 1.0f) / (norm);
		if (thread_grp_idx > 0)
			sub[gb_index_x*n + thread_grp_idx + i] = ((float)thread_grp_idx + 1.0f) / (norm * 3);
		else
			sub[gb_index_x*n + thread_grp_idx + i] = 0.0f;
	}
}

// Thomas resolution for tridiagonal symmetric matrices
__global__ void thom_sym_k(float *sub, float *diag, float *Y, int n){
	// The global memory access index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	//Local variables for easier access and less calculation
	int j;
	int first = idx * n, first_c = first + 1;
	int cur_first, cur_first_c;
	int n_first = first + n - 1, n_first_c = first_c + n - 2;

	//First forward substitution
	float d = diag[first];
	Cp[first_c] = sub[first_c] / d;
	Y[first] = Y[first] / d;

	//Decomposition and forward substitution.
	for (j=1;j<n-1;j++) {
		cur_first = first + j;
		cur_first_c = first_c + j;
		Cp[cur_first_c] = sub[cur_first_c] / (diag[cur_first] - sub[cur_first_c - 1]*Cp[cur_first_c - 1]);
		Y[cur_first] = (Y[cur_first] - sub[cur_first_c - 1]*Y[cur_first - 1]) / (diag[cur_first] - sub[cur_first_c - 1]*Cp[cur_first_c - 1]);
	}

	// One more iteration for Y (j=n-1)
	Y[n_first] = (Y[n_first] - sub[n_first_c]*Y[n_first - 1]) / (diag[n_first] - sub[n_first_c]*Cp[n_first_c]);

	for (j=(n-2);j>=0;j--){ //Backsubstitution.
		cur_first = first + j;
		cur_first_c = first_c + j;
		Y[cur_first] = Y[cur_first] - Cp[cur_first_c]*Y[cur_first + 1];
	}

}

__global__ void pcr_sym_k(float *sub, float *diag, float *Y, float *X, int n){
	// Identifies the thread working within a group
	int thread_grp_idx = threadIdx.x%n;
	// Identifies the data concerned by the computations
	int data_range = (threadIdx.x - thread_grp_idx) / n;
	// The global memory access index
	int gb_index_x = data_range + blockIdx.x*(blockDim.x / n);

	//Local variables
	int i, j;
	int sh_range = 5 * data_range * n, s = 1;
	int right_idx, left_idx;
	int index = gb_index_x * n + thread_grp_idx;
	float new_up_diag, new_diag, new_low_diag, new_res;
	float tmp1, tmp2, tmp3;

	// Shared memory
	extern __shared__ float sh_mem[];

	//Pointers to shared memory
	float * sh_upper_diag = (float *)&sh_mem[sh_range];
	float * sh_diag = (float *)&sh_upper_diag[n];
	float * sh_lower_diag = (float *)&sh_diag[n];
	float * sh_res = (float *)&sh_lower_diag[n];
	float * sh_sol = (float *)&sh_res[n];

	sh_upper_diag[thread_grp_idx] = sub[index]; //sh_upper_diag will contain the values of the upper diagonal for the current group
	sh_diag[thread_grp_idx] = diag[index]; //sh_diag will contain the values of the diagonal for the current group
	sh_lower_diag[thread_grp_idx] = sub[index + 1]; //sh_lower_diag will contain the values of the lower diagonal for the current group, 
	//meaning the sub diagonal from it's second index
	sh_res[thread_grp_idx] = Y[index]; //sh_res will contain the values of the result vector, the Y vector
	sh_sol[thread_grp_idx] = X[index]; //sh_sol will contain the values of the solution vector, the X vector

	__syncthreads();


	for (i = 0; i < (int)log2((float)n) + 1; i++) {

		j = thread_grp_idx;

		//Calculate the right and left index of the current cell
		right_idx = j + s;
		right_idx = right_idx & (n - 1);
		left_idx = j - s;
		left_idx = left_idx & (n - 1);

		//Calculate the pivots
		tmp1 = sh_upper_diag[j] / sh_diag[left_idx];
		tmp2 = sh_lower_diag[j] / sh_diag[right_idx];

		//Calculate the new values of the different component, modified by the pivots
		new_diag = sh_diag[j] - sh_lower_diag[left_idx] * tmp1 - sh_upper_diag[right_idx] * tmp2;
		new_res = sh_res[j] - sh_res[left_idx] * tmp1 - sh_res[right_idx] * tmp2;
		new_up_diag = -sh_upper_diag[left_idx] * tmp1;
		new_low_diag = -sh_lower_diag[right_idx] * tmp2;

		__syncthreads();

		//Permutate the current values with the new one, modified with pivot
		sh_diag[j] = new_diag;
		sh_res[j] = new_res;
		sh_upper_diag[j] = new_up_diag;
		sh_lower_diag[j] = new_low_diag;

		s <<= 1;
		__syncthreads();

	}

	if (thread_grp_idx < s){
		int addr = thread_grp_idx;
		tmp3 = sh_diag[addr] * sh_diag[addr] - sh_lower_diag[addr] * sh_upper_diag[addr];
		sh_sol[addr] = (sh_diag[addr] * sh_res[addr] - sh_lower_diag[addr] * sh_res[addr]) / tmp3;
	}

	__syncthreads();

	//Move the final solution to the X vector
	X[index] = sh_sol[thread_grp_idx];
}

int main() {

	int i, j;

	// Dim is the rank of the matrix

	for (int Dim = 2; Dim <= 1024; Dim *= 2) {

		// The number of blocks
		int NB = M / Dim;

		// The number of matrices to invert
		int size = NB;

		int minTB = 1024 / Dim;

		// The diagonal elements
		float *diag, *diagGPU;
		// The subdiagonal elements
		float *sub, *subGPU;
		// The system vector
		float *Y, *YGPU;
		float *X, *XGPU;

		float TimerV;					// GPU timer instructions
		cudaEvent_t start, stop;		// GPU timer instructions
		cudaEventCreate(&start);		// GPU timer instructions
		cudaEventCreate(&stop);			// GPU timer instructions

		// Memory allocation
		diag = (float *)calloc(size*Dim, sizeof(float));
		sub = (float *)calloc(size*Dim, sizeof(float));
		cudaMalloc(&diagGPU, size*Dim * sizeof(float));
		cudaMalloc(&subGPU, size*Dim * sizeof(float));

		X = (float *)calloc(size*Dim, sizeof(float));
		Y = (float *)calloc(size*Dim, sizeof(float));
		cudaMalloc(&XGPU, size*Dim * sizeof(float));
		cudaMalloc(&YGPU, size*Dim * sizeof(float));

		//Result vector
		for (i = 0; i < size; i++) {
			for (j = 0; j < Dim; j++) {
				Y[j + i * Dim] = 0.5f*j;
			}
		}
		cudaMemcpy(YGPU, Y, size*Dim * sizeof(float), cudaMemcpyHostToDevice);

		// Tridiagonal elements
		for (i = 0; i*Dim*NB < M; i++) {
			Tri_k << <NB, Dim >> > (subGPU, diagGPU, 10.0f, i*Dim*NB, Dim, Dim*NB);
		}

		// Resolution part
		cudaEventRecord(start, 0);

		/*
		Pour tester l'algorithme de Thomas, il faut décommenter les deux lignes :
		thom_sym_k << <NB / 256, 256 >> > (subGPU, diagGPU, YGPU, Dim);
		cudaMemcpy(X, YGPU, size*Dim * sizeof(float), cudaMemcpyDeviceToHost);
		printf("To solve %ld matrixes of size %ld, using the Thomas algorithm, the time taken was : %f ms\n", size, Dim, TimerV);

		commenter les lignes :
		pcr_sym_k << <NB / minTB, Dim*minTB, 5 * minTB*Dim * sizeof(float) >> > (subGPU, diagGPU, YGPU, XGPU, Dim);
		cudaMemcpy(X, XGPU, size*Dim * sizeof(float), cudaMemcpyDeviceToHost);
		printf("To solve %ld matrixes of size %ld, using the PCR algorithm, the time taken was : %f ms\n", size, Dim, TimerV);

		et
		Pour tester PCR, il suffit de faire l'inverse.
		*/

		thom_sym_k << <NB / 256, 256 >> > (subGPU, diagGPU, YGPU, Dim);
		//pcr_sym_k << <NB / minTB, Dim*minTB, 5 * minTB*Dim * sizeof(float) >> > (subGPU, diagGPU, YGPU, XGPU, Dim);


		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&TimerV, start, stop);

		cudaMemcpy(X, YGPU, size*Dim * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(X, XGPU, size*Dim * sizeof(float), cudaMemcpyDeviceToHost);

		/*for (i = 0; i < size; i++) {
			if (i == 573) {
				for (j = 0; j < Dim; j++) {
					printf("%.5e, ", X[j + i * Dim]);
				}
			}
		}*/

		printf("To solve %ld matrixes of size %ld, using the Thomas algorithm, the time taken was : %f ms\n", size, Dim, TimerV);
		//printf("To solve %ld matrixes of size %ld, using the PCR algorithm, the time taken was : %f ms\n", size, Dim, TimerV);


		// Memory free for other arrays
		free(diag);
		cudaFree(diagGPU);
		free(sub);
		cudaFree(subGPU);
		free(X);
		cudaFree(XGPU);
		free(Y);
		cudaFree(YGPU);

		cudaEventDestroy(start);		// GPU timer instructions
		cudaEventDestroy(stop);			// GPU timer instructions
	}
	return 0;
}