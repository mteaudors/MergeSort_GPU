
#include "cuda_header.h"

__device__ int A_diag[D];
__device__ int B_diag[D];


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


__global__ void mergeSmallBatch_k(int *AB, int length_A, int length_B, int *M, int nb_merge) {
	int nb_threads = gridDim.x * blockDim.x,
		gtidx = threadIdx.x + blockIdx.x * blockDim.x,
		d = length_A+length_B;
	
	int t, tidx, qt, gbx, begin_M, begin_A, begin_B;

	duo K, P, Q;

	while(gtidx < nb_merge*d) {
		
		t = gtidx % blockDim.x,                   		// index of the thread inside a block : 0 -> blockDim.x - 1
		tidx = t%d,                             		// index of the thread in its corresponding final array : 0 -> d-1
		qt = (t-tidx)/d,                        		// index of the group of the thread inside a block : 0 -> (blockDim.x/d)-1
		gbx = (blockDim.x/d)*blockIdx.x + qt,   		// index of the group of the thread among all the blocks : 0 -> (blockDim.x/d)*gridDim.x - 1
		begin_M = gbx*d,                        		// index of the first element of M
		begin_A = gbx*d,                 				// index of the first element of A
		begin_B = begin_A + length_A;                 	// index of the first element of B

		if (tidx > length_A) {
				K.x = tidx - length_A;
				K.y = length_A;
				P.x = length_A;
				P.y = tidx - length_A;
		}else {
				K.x = 0;
				K.y = tidx;
				P.x = tidx;
				P.y = 0;
		}
		while (true) {
				int offset = abs(K.y - P.y) / 2;
				Q.x = K.x + offset;
				Q.y = K.y - offset;
				if (Q.y >= 0 && Q.x <= length_B && (Q.y == length_A || Q.x == 0 || AB[begin_A + Q.y] > AB[begin_B + Q.x - 1])) {
						if (Q.x == length_B || Q.y == 0 || AB[begin_A + Q.y - 1] <= AB[begin_B + Q.x]) {
								if (Q.y < length_A && (Q.x == length_B || AB[begin_A + Q.y] <= AB[begin_B + Q.x])) {
										M[begin_M + tidx] = AB[begin_A + Q.y];
								}else {
										M[begin_M + tidx] = AB[begin_B + Q.x];
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
		gtidx += nb_threads;
	}
}

/**
	Find the intersection between one diagonal and the merge path.
*/
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B, int start_diag) {
	int nb_threads = gridDim.x * blockDim.x;
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int length = (length_A+length_B)/nb_threads;
	

	duo K, P, Q;

	while(tidx<(length_A+length_B)) {
		int index_diag = start_diag + tidx*length;

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
		tidx += nb_threads;
	}
	
}

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M, int start_diag) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;

	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	
	while(tidx<(length_A+length_B)) {
		int start_M = start_diag + tidx*length;

		for(int k=0 ; k<length ; ++k) {
				int i = start_M + k;
			if (A_diag[i] < length_A && (B_diag[i] == length_B || A[A_diag[i]] <= B[B_diag[i]])) {
				M[i] = A[A_diag[i]];
			} else {
				M[i] = B[B_diag[i]];
			}
		}
		tidx += nb_threads;
	}
}


void mergeSortGPU (int *M , int length, float *timer) {
    float timer_iter;
    int *M_dev, *M_dev_copy;
    int merge_size = 2;
    cudaEvent_t start, stop, start_iter, stop_iter;

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventCreate(&start_iter));
    testCUDA(cudaEventCreate(&stop_iter));

    testCUDA(cudaMalloc((void**)&M_dev , D*sizeof(int)));
    testCUDA(cudaMalloc((void**)&M_dev_copy , D*sizeof(int)));
    testCUDA(cudaMemcpy(M_dev, M,D*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA(cudaEventRecord(start, 0));
    
    while(merge_size <= pow(2,ceil(log2(length)))) {
	
    	testCUDA(cudaEventRecord(start_iter,0));

		testCUDA(cudaMemcpy(M_dev_copy, M_dev, D * sizeof(int), cudaMemcpyDeviceToDevice));
	
		if(merge_size <= BATCH_THRESHOLD) {
			int block_size =  min(length, ((int)(1024/merge_size))*merge_size);
			int nb_block = (length + block_size - 1)/block_size;
			mergeSmallBatch_k<<<nb_block,block_size>>>(M_dev_copy, merge_size/2, merge_size/2, M_dev, length/merge_size);

			if(length%merge_size) {
				int merge_size_last = length%merge_size;
				int block_size_last = (merge_size_last > 1024) ? 1024 : merge_size_last;
				int nb_block_last = (merge_size_last + block_size_last - 1)/block_size_last;
				
				if(merge_size_last > merge_size/2) 
					mergeSmall_k<<<nb_block_last,block_size_last>>>(M_dev_copy + ((int)(length/merge_size))*merge_size, merge_size/2, M_dev_copy + ((int)(length/merge_size))*merge_size + (merge_size/2), merge_size_last-(merge_size/2), M_dev + ((int)(length/merge_size))*merge_size);
			}
		}
		else {
			int iter = (length+merge_size-1)/merge_size;
			int block_size = (merge_size > 1024) ? 1024 : merge_size;
			int nb_block = (merge_size+block_size-1)/block_size;
			
			for (int k = 0; k < iter; ++k) {
				if(k<(length/merge_size))
					pathBig_k << <nb_block,block_size>> > (M_dev + k * merge_size, merge_size / 2, M_dev + (2 * k + 1)*(merge_size / 2), merge_size / 2, merge_size*k);
				else {
					int merge_size_last = length%merge_size;
					int block_size_last = (merge_size_last > 1024) ? 1024 : merge_size_last;
					int nb_block_last = (merge_size_last + block_size_last - 1)/block_size_last;
					
					while(nb_block_last*block_size_last > merge_size_last) block_size_last /= 2;
					
					if(merge_size_last > merge_size/2) 
						pathBig_k << <nb_block_last, block_size_last >> > (M_dev + k * merge_size, merge_size / 2,M_dev + k * merge_size + merge_size/2, merge_size_last - (merge_size/2), merge_size*k);
				}
			}
			testCUDA(cudaDeviceSynchronize());
				
			for(int k=0 ; k<iter ; ++k) {
				if(k<(length/merge_size))
					mergeBig_k<<<nb_block,block_size>>>(M_dev_copy+k*merge_size, merge_size/2, M_dev_copy+(2*k+1)*(merge_size/2), merge_size/2, M_dev, merge_size*k);
				else {
					int merge_size_last = length%merge_size;
					int block_size_last = (merge_size_last > 1024) ? 1024 : merge_size_last;
					int nb_block_last = (merge_size_last + block_size_last - 1)/block_size_last;
					
					while(nb_block_last*block_size_last > merge_size_last) block_size_last /= 2;
		
					if(merge_size_last > merge_size/2)
						mergeBig_k << <nb_block_last, block_size_last >> > (M_dev_copy + k * merge_size, merge_size / 2,M_dev_copy + k * merge_size + merge_size/2, merge_size_last - (merge_size/2),  M_dev, merge_size*k);
				
				}
		
			}
		}

		testCUDA(cudaDeviceSynchronize());
		testCUDA(cudaEventRecord(stop_iter,0));
		testCUDA(cudaEventSynchronize(stop_iter));
		testCUDA(cudaEventElapsedTime(&timer_iter,start_iter, stop_iter));
		printf("MergeSize = %7d\t\tDuration : %f ms\n",merge_size,timer_iter);
		merge_size *= 2;

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

    check_array_sorted(M,D,"M");
    printf("===== Total time : %f ms =====\n", TimerAdd);
    free(M);
}
