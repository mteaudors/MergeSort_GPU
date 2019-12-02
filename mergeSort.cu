
#include "cuda_header.h"

__device__ int A_diag[D];
__device__ int B_diag[D];

/**
	Find the intersection between one diagonal and the merge path.
*/
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B, int start_diag) {
	int nb_threads = gridDim.x * blockDim.x;
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int length = (length_A+length_B)/nb_threads;
	int index_diag = start_diag + tidx*length;

        duo K, P, Q;
	for(int k=0 ; k<length; ++k) {
		int i = tidx*length+k;
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
					A_diag[index_diag+k] = Q.y;
					B_diag[index_diag+k] = Q.x;
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
}

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M, int start_diag) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = start_diag + tidx*length;

	for(int k=0 ; k<length ; ++k) {
        	int i = start_M + k;
		if (A_diag[i] < length_A && (B_diag[i] == length_B || A[A_diag[i]] <= B[B_diag[i]])) {
            		M[i] = A[A_diag[i]];
		} else {
			M[i] = B[B_diag[i]];
		}
	}
}


void mergeSortGPU (int *M , int length, float *timer) {
    int *M_dev, *M_dev_copy;
	int mergeSize = 2;
	cudaEvent_t start, stop;

	testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));

    testCUDA(cudaMalloc((void**)&M_dev , D*sizeof(int)));
    testCUDA(cudaMalloc((void**)&M_dev_copy , D*sizeof(int)));
    testCUDA(cudaMemcpy(M_dev, M,D*sizeof(int), cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);
    
    while(mergeSize <= length) {
	
	testCUDA(cudaMemcpy(M_dev_copy, M_dev, D * sizeof(int), cudaMemcpyDeviceToDevice));
	int block_size = (mergeSize > 1024) ? 1024 : mergeSize;
	int nb_block = (mergeSize+block_size-1)/block_size;
	for (int k = 0; k < (length / mergeSize); ++k) {
		pathBig_k << <nb_block,block_size>> > (M_dev + k * mergeSize, mergeSize / 2, M_dev + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, mergeSize*k);
	}
	testCUDA(cudaDeviceSynchronize());
        for(int k=0 ; k<length/mergeSize ; ++k) {
		mergeBig_k<<<nb_block,block_size>>>(M_dev_copy+k*mergeSize, mergeSize/2, M_dev_copy+(2*k+1)*(mergeSize/2), mergeSize/2, M_dev, mergeSize*k);

        }
	testCUDA(cudaDeviceSynchronize());
	printf("Merge size : %d\tBlock size : %d\tGrid size : %d\n",mergeSize,block_size,nb_block);
	mergeSize *= 2;

    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timer, start, stop);
    
    testCUDA(cudaMemcpy(M, M_dev,D*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaFree(M_dev));
    testCUDA(cudaFree(M_dev_copy));

	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
}

int main(int argc , char *argv[]) {
    // initialize random seed
    srand(time(0));
	float TimerAdd = 0;

    int* M = generate_unsorted_array(D);
    //print_unsorted_array(M , D , "M");

    mergeSortGPU(M,D, &TimerAdd);

    print_array(M,D,"M");
    printf("Time elapsed : %f ms\n", TimerAdd);
    free(M);
}
