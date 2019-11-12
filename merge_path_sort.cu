#include "cuda_header.h"


__device__ int A_diag[SIZE_M];
__device__ int B_diag[SIZE_M];

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
}

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i*(length_A+length_B)/ nb_threads;

	__shared__ int A_index[SIZE_M];
	__shared__ int B_index[SIZE_M];

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
}


/*
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B) {
	int nb_threads = gridDim.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int index_diag = i * (length_A + length_B) / nb_threads;
	
	int a_top = (index_diag > length_A) ? length_A : index_diag;
	int b_top = (index_diag > length_A) ? index_diag - length_A : 0;
	int a_bot = b_top;
	printf("thread %d from block %d working on diag %d ==> a_top : %d a_bottom : %d\n",i,blockIdx.x,index_diag,a_top,a_bot);

	while (true) {
		int offset = abs(a_top - a_bot) / 2;
		int a_i = a_top - offset;
		int b_i = b_top + offset;
		if (A[a_i] > B[b_i - 1]) {
			if(index_diag==0) printf("First condition OK\n");
			if (A[a_i - 1] > B[b_i]) {
				A_diag[i] = a_i;
				B_diag[i] = b_i;
				printf("found thread %d in block %d\n",i,blockIdx.x);
				break;
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
	
	printf("le thread %d fait les valeurs %d jusqu'� %d, donc [%d,%d] jusqu'� [%d,%d]\n", 
		index, index * length, (index + 1) * length-1, A_diag[index * length], B_diag[index * length], A_diag[(index + 1) * length - 1], B_diag[(index + 1) * length - 1]);

	
	//Malgre la reponse du prof, je vois toujours pas l'utilite de faire ca, ramener en shared puis affter ...

	// Je suis pas sur de cette boucle pour ramener, parce que j'ai pas lu l'article, et je ne sais donc pas exactement ce que contient A_diag et B_diag. 
	for (int i = 0; i < length; ++i) {
		shared_A[threadIdx.x * length + i] = A_diag[index * length + i];
		shared_B[threadIdx.x * length + i] = B_diag[index * length + i];
	}

	__syncthreads(); // wait for each thread to copy its element

	// Encore moins sur qu'avant, je ne sais pas s'il y a des conditions � mettre ou autre chose. 
	for (int i = 0; i < length; ++i) {
		int temp_a = shared_A[threadIdx.x * length + i];
		int temp_b = shared_B[threadIdx.x * length + i];
		M[index * length + i] = (temp_a < temp_b) ? temp_a : temp_b;
	}
}
*/

int main()
{   
    // initialize random seed
    srand(time(0));
	testCUDA(cudaDeviceReset());
    
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

	// Launch path kernel on GPU
	
	printf("Launch %d blocks of %d threads\n",GRIDSIZE(SIZE_M),BLOCKSIZE);
	pathBig_k<<<GRIDSIZE(SIZE_M), BLOCKSIZE>>>(dev_a, SIZE_A, dev_b, SIZE_B);
	
	// Wait until work is done
	cudaDeviceSynchronize();

	// Launch merge kernel on GPU
	mergeBig_k <<<1 , BLOCKSIZE/4>> > (dev_a, SIZE_A, dev_b, SIZE_B, dev_m);

	// Wait until work is done
	cudaDeviceSynchronize();

	// Copy result from GPU RAM into CPU RAM
	testCUDA(cudaMemcpy(M, dev_m, LENGTH_M, cudaMemcpyDeviceToHost));

	// print arrays
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

	testCUDA(cudaGetLastError());
	testCUDA(cudaDeviceReset());

    return 0;
}