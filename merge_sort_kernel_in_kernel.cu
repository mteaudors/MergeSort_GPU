
#include "cuda_header.h"

__device__ int A_diag[D];
__device__ int B_diag[D];

/**
	Find the intersection between one diagonal and the merge path.
	�a sert � r de mettre + de thread que d'diag.
*/
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B, int start_diag) {
	int nb_threads = gridDim.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int index_diag = (i + start_diag) * (length_A + length_B) / nb_threads;

	//printf("Writting in A_diag at index : %d\n",index_diag);

	duo K, P, Q;
	if (i > length_A) {
		K.x = i - length_A;
		K.y = length_A;
		P.x = length_A;
		P.y = i - length_A;
	}
	else {
		K.x = 0;
		K.y = i;
		P.x = i;
		P.y = 0;
	}
	while (true) {
		int offset = abs(K.y - P.y) / 2;
		Q.x = K.x + offset;
		Q.y = K.y - offset;
		if (Q.y >= 0 && Q.x <= length_B && (Q.y == length_A || Q.x == 0 || A[Q.y] > B[Q.x - 1])) {
			if (Q.x == length_B || Q.y == 0 || A[Q.y - 1] <= B[Q.x]) {
				A_diag[index_diag] = Q.y;
				B_diag[index_diag] = Q.x;
				break;
			}
			else {
				K.x = Q.x + 1;
				K.y = Q.y - 1;
			}
		}
		else {
			P.x = Q.x - 1;
			P.y = Q.y + 1;
		}
	}
}

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M, int start_diag) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i * (length_A + length_B) / nb_threads;

	for (int k = 0; k < length; ++k) {
		int i = start_diag + start_M + k;
		if (A_diag[i] < length_A && (B_diag[i] == length_B || A[A_diag[i]] <= B[B_diag[i]])) {
			M[start_M + k] = A[A_diag[i]];
		}
		else {
			M[start_M + k] = B[B_diag[i]];
		}
	}
}

__device__ void tempPathBig_k(int *A, int length_A, int *B, int length_B, int start_diag) {
	int nb_threads = gridDim.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int index_diag = (i + start_diag) * (length_A + length_B) / nb_threads;

	//printf("Writting in A_diag at index : %d\n",index_diag);

	duo K, P, Q;
	if (i > length_A) {
		K.x = i - length_A;
		K.y = length_A;
		P.x = length_A;
		P.y = i - length_A;
	}
	else {
		K.x = 0;
		K.y = i;
		P.x = i;
		P.y = 0;
	}
	while (true) {
		int offset = abs(K.y - P.y) / 2;
		Q.x = K.x + offset;
		Q.y = K.y - offset;
		if (Q.y >= 0 && Q.x <= length_B && (Q.y == length_A || Q.x == 0 || A[Q.y] > B[Q.x - 1])) {
			if (Q.x == length_B || Q.y == 0 || A[Q.y - 1] <= B[Q.x]) {
				A_diag[index_diag] = Q.y;
				B_diag[index_diag] = Q.x;
				break;
			}
			else {
				K.x = Q.x + 1;
				K.y = Q.y - 1;
			}
		}
		else {
			P.x = Q.x - 1;
			P.y = Q.y + 1;
		}
	}
}

__device__ void tempMergeBig_k(int *A, int length_A, int *B, int length_B, int* M, int start_diag) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i * (length_A + length_B) / nb_threads;

	for (int k = 0; k < length; ++k) {
		int i = start_diag + start_M + k;
		if (A_diag[i] < length_A && (B_diag[i] == length_B || A[A_diag[i]] <= B[B_diag[i]])) {
			M[start_M + k] = A[A_diag[i]];
		}
		else {
			M[start_M + k] = B[B_diag[i]];
		}
	}
}

__device__ void threadBlockDeviceSynchronize(void) {
	__syncthreads();
	if (threadIdx.x == 0)
		cudaDeviceSynchronize();
	__syncthreads();
}

__global__ void merge_k(int *M_dev_A, int *M_dev_copy_A, int length_A, int *M_dev_B, int *M_dev_copy_B, int length_B, int *M, int start_diag) {
	tempPathBig_k(M_dev_A, length_A, M_dev_B, length_B, start_diag);
	tempMergeBig_k(M_dev_copy_A, length_A, M_dev_copy_B, length_B, M, start_diag);
}

/*
tidx = indice du thread dans son groupe de tableaux � trier, genre 32 tableaux de 32 cases � trier dans un block
Qt = indice du groupe de travail du thread actuel
gbx = indice du thread pour acc�der � la m�moire globale, indice d'un groupe de thread dans l'ensemble des threads pour pouvoir acc�der � la RAM
*/


void mergeSortGPU(int *M, int length, float *timer) {
	int *M_dev, *M_dev_copy;
	int mergeSize = 2;
	cudaEvent_t start, stop;

	testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));

	testCUDA(cudaMalloc((void**)&M_dev, D * sizeof(int)));
	testCUDA(cudaMalloc((void**)&M_dev_copy, D * sizeof(int)));
	testCUDA(cudaMemcpy(M_dev, M, D * sizeof(int), cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);

	while (mergeSize <= length) {
		testCUDA(cudaMemcpy(M_dev_copy, M_dev, D * sizeof(int), cudaMemcpyDeviceToDevice));
		for (int k = 0; k < (length / mergeSize); ++k) {
			pathBig_k << <1, mergeSize >> > (M_dev + k * mergeSize, mergeSize / 2, M_dev + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, mergeSize*k);
		}
		testCUDA(cudaDeviceSynchronize());
		for (int k = 0; k < length / mergeSize; ++k) {
			mergeBig_k << <1, mergeSize >> > (M_dev_copy + k * mergeSize, mergeSize / 2, M_dev_copy + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, M_dev + k * mergeSize, mergeSize*k);
		}
		testCUDA(cudaDeviceSynchronize());
		mergeSize *= 2;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timer, start, stop);

	testCUDA(cudaMemcpy(M, M_dev, D * sizeof(int), cudaMemcpyDeviceToHost));
	testCUDA(cudaFree(M_dev));
	testCUDA(cudaFree(M_dev_copy));

	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
}

void mergeSortGPUprim(int *M, int length, float *timer) {
	int *M_dev, *M_dev_copy;
	int mergeSize = 2;
	cudaEvent_t start, stop;

	testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));

	testCUDA(cudaMalloc((void**)&M_dev, D * sizeof(int)));
	testCUDA(cudaMalloc((void**)&M_dev_copy, D * sizeof(int)));
	testCUDA(cudaMemcpy(M_dev, M, D * sizeof(int), cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);

	while (mergeSize <= length) {
		testCUDA(cudaMemcpy(M_dev_copy, M_dev, D * sizeof(int), cudaMemcpyDeviceToDevice));
		for (int k = 0; k < (length / mergeSize); ++k) {
			merge_k << <1, mergeSize >> > (M_dev + k * mergeSize, M_dev_copy + k * mergeSize, mergeSize / 2,
				M_dev + (2 * k + 1)*(mergeSize / 2), M_dev_copy + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, M_dev + k * mergeSize, mergeSize*k);
		}
		cudaDeviceSynchronize();
		mergeSize *= 2;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timer, start, stop);

	testCUDA(cudaMemcpy(M, M_dev, D * sizeof(int), cudaMemcpyDeviceToHost));
	testCUDA(cudaFree(M_dev));
	testCUDA(cudaFree(M_dev_copy));

	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
}


int main(int argc, char *argv[]) {
	// initialize random seed
	srand(time(0));
	float TimerAdd = 0;
	float TimerAdd2 = 0;

	int* M = generate_unsorted_array(D);
	int* M_prim = generate_unsorted_array(D);
	print_unsorted_array(M, D, "M");
	print_unsorted_array(M_prim, D, "M'");

	mergeSortGPU(M, D, &TimerAdd);
	mergeSortGPUprim(M_prim, D, &TimerAdd2);

	print_array(M, D, "M");
	printf("Time elapsed : %f ms\n", TimerAdd);
	printf("Time elapsed2 : %f ms\n", TimerAdd2);
	free(M);
}