#include "cuda_header.h"

__global__ void mergeSmall_k(int *A, int length_A, int *B, int length_B, int *M) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
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
                                if (Q.y < length_A && (Q.x == length_B || A[Q.y] <= B[Q.x])) {
                                        M[i] = A[Q.y];
                                }else {
                                        M[i] = B[Q.x];
								}
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

int main(int argc, char *argv[]) {
    // Allocate CPU buffers for three vectors (two input, one output).
	int* A = generate_array(SIZE_A);
	int* B = generate_array(SIZE_B);
	int* M = (int*)malloc(LENGTH_M);

	// Declare GPU buffers
	int *dev_a = nullptr;
	int *dev_b = nullptr;
	int *dev_m = nullptr;

	// Allocate GPU buffers for three vectors (two input, one output).
	testCUDA(cudaMalloc((void**)&dev_a, LENGTH_A));
	testCUDA(cudaMalloc((void**)&dev_b, LENGTH_B));
	testCUDA(cudaMalloc((void**)&dev_m, LENGTH_M));

	// Copy input vectors from host memory to GPU buffers.
	testCUDA(cudaMemcpy(dev_a, A, LENGTH_A, cudaMemcpyHostToDevice));
	testCUDA(cudaMemcpy(dev_b, B, LENGTH_B, cudaMemcpyHostToDevice));

	// Launch merge kernel on GPU
	mergeSmall_k << <1, SIZE_M >> > (dev_a, SIZE_A, dev_b, SIZE_B, dev_m);

	// Wait until work is done
	cudaDeviceSynchronize();

	// Copy result from GPU RAM into CPU RAM
	testCUDA(cudaMemcpy(M, dev_m, LENGTH_M, cudaMemcpyDeviceToHost));

	print_array(A, SIZE_A, "A");
	print_array(B, SIZE_B, "B");
	print_array(M, SIZE_M, "M");

	// Free both CPU and GPU memory allocated
	testCUDA(cudaFree(dev_m));
	testCUDA(cudaFree(dev_b));
	testCUDA(cudaFree(dev_a));
	free(M);
	free(B);
	free(A);
}