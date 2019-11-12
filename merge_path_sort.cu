#include "cuda_header.h"

void merge_sort(int *A, int length_a, int *B, int length_b, int *M, int length_m) {
	int i = 0, j = 0;

	while ((i + j) < length_m) {
		if (j >= length_b || (i < length_a && A[i] < B[j])) {
			M[i + j] = A[i++];
		} else {
			M[i + j] = B[j++];
		}
	}
}

void sequential_merge_sort() {
	int *A = generate_array(LENGTH_A);
	print_array(A, SIZE_A, "A");
	int *B = generate_array(LENGTH_B);
	print_array(B, SIZE_B, "B");
	int *M = (int*) malloc(LENGTH_M);
	
	merge_sort(A, SIZE_A, B, SIZE_B, M, SIZE_M);
	print_array(M, SIZE_M, "M");
	merge_sort(B, SIZE_B, A, SIZE_A, M, SIZE_M);
	print_array(M, SIZE_M, "M");

	free(M);
	free(B);
	free(A);
}

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

void question_1() {

	// Allocate CPU buffers for three vectors (two input, one output).
	int* A = generate_array(LENGTH_A);
	int* B = generate_array(LENGTH_B);
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

__device__ int A_diag[SIZE_M];
__device__ int B_diag[SIZE_M];

/* Simple application de l'algorithme donné dans l'article. */
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B) {
	int nb_threads = gridDim.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int index = i * (length_A + length_B) / nb_threads;
	int a_top = (index > length_A) ? length_A : index;
	int b_top = (index > length_A) ? index - length_A : 0;
	int a_bot = b_top;
	while (true) {
		int offset = (a_top - a_bot) / 2;
		int a_i = a_top - offset;
		int b_i = b_top + offset;
		if (A[a_i] > B[b_i - 1]) {
			if (A[a_i - 1] > B[b_i]) {
				A_diag[i] = a_i;
				B_diag[i] = b_i;
			}
			else {
				a_top = a_i - 1;
				b_top = b_i + 1;
			}
		}
		else {
			a_bot = a_i + 1;
		}
	}
}

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	__shared__ float shared_A[1024];
	__shared__ float shared_B[1024];
	printf("le thread %d fait les valeurs %d jusqu'à %d, donc [%d,%d] jusqu'à [%d,%d]\n", 
		index, index * length, (index + 1) * length-1, A_diag[index * length], B_diag[index * length], A_diag[(index + 1) * length - 1], B_diag[(index + 1) * length - 1]);

	/*
	Malgré la réponse du prof, je vois toujours pas l'utilité de faire ça, ramener en shared puis affter à M...
	*/

	/* Je suis pas sur de cette boucle pour ramener, parce que j'ai pas lu l'article, et je ne sais donc pas exactement ce que contient A_diag et B_diag. */
	for (int i = 0; i < length; ++i) {
		shared_A[threadIdx.x * length + i] = A_diag[index * length + i];
		shared_B[threadIdx.x * length + i] = B_diag[index * length + i];
	}

	__syncthreads();// wait for each thread to copy its element

	/* Encore moins sur qu'avant, je ne sais pas s'il y a des conditions à mettre ou autre chose. */
	for (int i = 0; i < length; ++i) {
		int temp_a = shared_A[threadIdx.x * length + i];
		int temp_b = shared_B[threadIdx.x * length + i];
		M[index * length + i] = (temp_a < temp_b) ? temp_a : temp_b;
	}
}

void question_2() {

	// Allocate CPU buffers for three vectors (two input, one output).
	int* A = generate_array(LENGTH_A);
	int* B = generate_array(LENGTH_B);
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

	// Launch path kernel on GPU
	pathBig_k<<<1, SIZE_M>>>(dev_a, SIZE_A, dev_b, SIZE_B);

	// Wait until work is done
	cudaDeviceSynchronize();

	// Launch merge kernel on GPU
	mergeBig_k << <1, SIZE_M >> > (dev_a, SIZE_A, dev_b, SIZE_B, dev_m);

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

int main()
{   
    // initialize random seed
    srand(time(0));
	testCUDA(cudaDeviceReset());
    
    //sequential_merge_sort();
    //question_1();
	question_2();
	testCUDA(cudaGetLastError());
	testCUDA(cudaDeviceReset());

    return 0;
}