
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
    float timer_iter;
    int *M_dev, *M_dev_copy;
    int mergeSize = 2;
    cudaEvent_t start, stop, start_iter, stop_iter;

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventCreate(&start_iter));
    testCUDA(cudaEventCreate(&stop_iter));

    testCUDA(cudaMalloc((void**)&M_dev , D*sizeof(int)));
    testCUDA(cudaMalloc((void**)&M_dev_copy , D*sizeof(int)));
    testCUDA(cudaMemcpy(M_dev, M,D*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA(cudaEventRecord(start, 0));
    
    while(mergeSize <= pow(2,ceil(log2(length)))) {
	
    	testCUDA(cudaEventRecord(start_iter,0));

	testCUDA(cudaMemcpy(M_dev_copy, M_dev, D * sizeof(int), cudaMemcpyDeviceToDevice));
	
	int iter = (length+mergeSize-1)/mergeSize;
	int block_size = (mergeSize > 1024) ? 1024 : mergeSize;
	int nb_block = (mergeSize+block_size-1)/block_size;
	
	for (int k = 0; k < iter; ++k) {
		if(k<(length/mergeSize))
			pathBig_k << <nb_block,block_size>> > (M_dev + k * mergeSize, mergeSize / 2, M_dev + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, mergeSize*k);
		else {
			int mergeSizeLast = length%mergeSize;
			int block_size_last = (mergeSizeLast > 1024) ? 1024 : mergeSizeLast;
			int nb_block_last = (mergeSizeLast + block_size_last - 1)/block_size_last;
			printf("MergeSize : %d\tMergeSizeLast %d\n",mergeSize,mergeSizeLast);
			
			while(nb_block_last*block_size_last > mergeSizeLast) block_size_last /= 2;
			
			printf("Launch %d blocks of %d threads\n",nb_block_last,block_size_last);
		       	if(mergeSizeLast > mergeSize/2) 
				pathBig_k << <nb_block_last, block_size_last >> > (M_dev + k * mergeSize, mergeSize / 2,M_dev + k * mergeSize + mergeSize/2, mergeSizeLast - (mergeSize/2), mergeSize*k);
		}
	}
	testCUDA(cudaDeviceSynchronize());
        
	for(int k=0 ; k<iter ; ++k) {
		if(k<(length/mergeSize))
			mergeBig_k<<<nb_block,block_size>>>(M_dev_copy+k*mergeSize, mergeSize/2, M_dev_copy+(2*k+1)*(mergeSize/2), mergeSize/2, M_dev, mergeSize*k);
		else {
			int mergeSizeLast = length%mergeSize;
			int block_size_last = (mergeSizeLast > 1024) ? 1024 : mergeSizeLast;
			int nb_block_last = (mergeSizeLast + block_size_last - 1)/block_size_last;
			
			while(nb_block_last*block_size_last > mergeSizeLast) block_size_last /= 2;

			if(mergeSizeLast > mergeSize/2)
				mergeBig_k << <nb_block_last, block_size_last >> > (M_dev_copy + k * mergeSize, mergeSize / 2,M_dev_copy + k * mergeSize + mergeSize/2, mergeSizeLast - (mergeSize/2),  M_dev, mergeSize*k);
		
		}

        }
	testCUDA(cudaDeviceSynchronize());
	testCUDA(cudaEventRecord(stop_iter,0));
	testCUDA(cudaEventSynchronize(stop_iter));
	testCUDA(cudaEventElapsedTime(&timer_iter,start_iter, stop_iter));
	printf("MergeSize = %d\t\tDuration : %f ms\n",mergeSize,timer_iter);
	mergeSize *= 2;

    }
    
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(timer, start, stop));
    
    testCUDA(cudaMemcpy(M, M_dev,D*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaFree(M_dev));
    testCUDA(cudaFree(M_dev_copy));

    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
    testCUDA(cudaEventDestroy(start_iter));
    testCUDA(cudaEventDestroy(stop_iter));

}

int main(int argc , char *argv[]) {
    // initialize random seed
    srand(time(0));
    float TimerAdd = 0;

    printf("Size of array : %d\n",D);

    int* M = generate_unsorted_array(D);
    //print_unsorted_array(M , D , "M");

    mergeSortGPU(M,D, &TimerAdd);

    print_array(M,D,"M");
    printf("===== Total time : %f ms =====\n", TimerAdd);
    free(M);
}
