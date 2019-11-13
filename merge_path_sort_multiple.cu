#include "cuda_header.h"


__device__ int A_diag[TOTAL_SIZE];
__device__ int B_diag[TOTAL_SIZE];

/**
	Find the intersection between one diagonal and the merge path.
*/
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B) {
		int nb_threads = gridDim.x * blockDim.x;
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int index_diag = i * (length_A + length_B) / nb_threads;

		duo K, P, Q;
        if (i > length_A) {
                K.x = i - length_A;
                K.y = length_A;
                P.x = length_A;
                P.y = i - length_A;
        }else {
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

		/*__syncthreads();

		if (i == 0) {
			for (int j = 0; j < (length_A + length_B); ++j) {
				printf("A_diag[%d] = %d, B_diag[%d] = %d\n", j, A_diag[j], j, B_diag[j]);
			}
		}*/
}

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i*(length_A+length_B)/ nb_threads;

	__shared__ int A_index[TOTAL_SIZE];
	__shared__ int B_index[TOTAL_SIZE];

	for(int k=0 ; k<length ; ++k) {
		A_index[start_M + k] = A_diag[start_M + k];
		B_index[start_M + k] = B_diag[start_M + k];
	}

	for(int k=0 ; k<length ; ++k) {
		int i = start_M + k;
		if (A_index[i] < length_A && (B_index[i] == length_B || A[A_index[i]] <= B[B_index[i]])) {
				M[i] = A[A_index[i]];
		} else {
				M[i] = B[B_index[i]];
		}
	}
	
	/*__syncthreads();

	if (i == 0) {
		for (int j = 0; j < (length_A + length_B); ++j) {
			printf("M[%d] = %d\n", j, M[j]);
		}
	}*/
}

int main()
{   
    // initialize random seed
    srand(time(0));
	testCUDA(cudaDeviceReset());

	int* array[ARRAY_NUMBER];
	for (int i = 0; i < ARRAY_NUMBER; ++i) {
		array[i] = generate_array(ARRAY_SIZES);
		print_array(array[i], ARRAY_SIZES, std::to_string(i));
	}

	int* M = (int*) malloc(TOTAL_LENGTH);

	// Declare GPU buffers
	int *dev_a = nullptr;
	int *dev_b = nullptr;
	int *dev_m = nullptr;

	// Allocate GPU buffers for three vectors (two input, one output).
	testCUDA(cudaMalloc((void**)&dev_a, ARRAY_LENGTH));

	for (int i = 0; i < ARRAY_NUMBER; ++i) {
		testCUDA(cudaMalloc((void**)&dev_b, i * ARRAY_LENGTH));
		testCUDA(cudaMalloc((void**)&dev_m, ARRAY_LENGTH + i * ARRAY_LENGTH));

		testCUDA(cudaMemcpy(dev_a, array[i], ARRAY_LENGTH, cudaMemcpyHostToDevice));
		testCUDA(cudaMemcpy(dev_b, M, i * ARRAY_LENGTH, cudaMemcpyHostToDevice));

		// Launch path kernel on GPU
		printf("Launch %lld blocks of %d threads\n", GRIDSIZE(ARRAY_SIZES + i * ARRAY_SIZES), BLOCKSIZE);
		pathBig_k << <GRIDSIZE(ARRAY_SIZES + i * ARRAY_SIZES), BLOCKSIZE >> > (dev_a, ARRAY_SIZES, dev_b, i * ARRAY_SIZES);

		// Wait until work is done
		testCUDA(cudaDeviceSynchronize());

		// Launch merge kernel on GPU
		mergeBig_k << <1, BLOCKSIZE / 4 >> > (dev_a, ARRAY_SIZES, dev_b, i * ARRAY_SIZES, dev_m);

		// Wait until work is done
		testCUDA(cudaDeviceSynchronize());

		// Copy result from GPU RAM into CPU RAM
		testCUDA(cudaMemcpy(M, dev_m, ARRAY_LENGTH + i * ARRAY_LENGTH, cudaMemcpyDeviceToHost));

		testCUDA(cudaFree(dev_m));
		testCUDA(cudaFree(dev_b));
	}

	// print arrays
	print_array(M, TOTAL_SIZE, "M");

	// Free both CPU and GPU memory allocated
	testCUDA(cudaFree(dev_a));
	for (int i = 0; i < ARRAY_NUMBER; ++i) {
		free(array[i]);
	}
	free(M);

	testCUDA(cudaGetLastError());
	testCUDA(cudaDeviceReset());

    return 0;
}