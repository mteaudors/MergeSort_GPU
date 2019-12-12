#include "cuda_header.h"

/*
	GPU kernel to merge, using a batch method, multiple couples of array (a,b).
*/
__global__ void mergeSmallBatch_k(int *A, int length_A, int *B, int length_B, int *M, int nb_merge) {
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
		begin_A = gbx*length_A,               				// index of the first element of A
		begin_B = gbx*length_B;  		               	// index of the first element of B

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
				if (Q.y >= 0 && Q.x <= length_B && (Q.y == length_A || Q.x == 0 || A[begin_A + Q.y] > B[begin_B + Q.x - 1])) {
						if (Q.x == length_B || Q.y == 0 || A[begin_A + Q.y - 1] <= B[begin_B + Q.x]) {
								if (Q.y < length_A && (Q.x == length_B || A[begin_A + Q.y] <= B[begin_B + Q.x])) {
										M[begin_M + tidx] = A[begin_A + Q.y];
								}else {
										M[begin_M + tidx] = B[begin_B + Q.x];
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


int main(int argc, char *argv[]) {

    // initialize random seed
    srand(time(0));
    testCUDA(cudaDeviceReset());
	cudaEvent_t start, stop;

	testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));

	float timer;

    /*  Number of threads per block.
        We want to launch N*SIZE_M threads to bind each thread with one element in the final array.
        We choose N*SIZE_M if it fits into one block. Otherwise we take the biggest multiple of SIZE_M <= 1024.
    */
    int tpb = min(N*SIZE_M, ((int)(1024/SIZE_M))*SIZE_M);    

    // To overwrite the default choice of tpb, just give it as the first program argument
    if(argc == 2) tpb = atoi(argv[1]); 

    // number of blocks 
    int blk = (N*SIZE_M + tpb - 1)/tpb;                         

    printf("------------------------------------------------------------------------------------------------------------------------------\n\n");
    printf("N : %d\n",N);
    printf("Size of A : %d\tSize of B : %d\tSize of M : %d\n",SIZE_A,SIZE_B,SIZE_M);
    printf("Minimum number of threads to launch : %d\n",N*SIZE_M);
    printf("Nb Blocks : %d\tNb threads/block : %d\t==>\t%d threads\n",blk, tpb, blk*tpb);
    printf("\n------------------------------------------------------------------------------------------------------------------------------\n");

    // Allocate CPU buffers for three vectors (two input, one output).
	int* A = generate_array_batch(N*SIZE_A, SIZE_A);
	int* B = generate_array_batch(N*SIZE_B, SIZE_B);

    	int* M = (int*)malloc(N*SIZE_M*sizeof(int));
   
	// Declare GPU buffers
	int *dev_a = nullptr;
	int *dev_b = nullptr;
	int *dev_m = nullptr;

	// Allocate GPU buffers for three vectors (two input, one output).
	testCUDA(cudaMalloc((void**)&dev_a, N*SIZE_A*sizeof(int)));
	testCUDA(cudaMalloc((void**)&dev_b, N*SIZE_B*sizeof(int)));
	testCUDA(cudaMalloc((void**)&dev_m, N*SIZE_M*sizeof(int)));

	// Copy input vectors from host memory to GPU buffers.
	testCUDA(cudaMemcpy(dev_a, A, N*SIZE_A*sizeof(int), cudaMemcpyHostToDevice));
	testCUDA(cudaMemcpy(dev_b, B, N*SIZE_B*sizeof(int), cudaMemcpyHostToDevice));

	testCUDA(cudaEventRecord(start, 0));
    // Launch merge kernel on GPU
	mergeSmallBatch_k << <blk, tpb >> > (dev_a, SIZE_A, dev_b, SIZE_B, dev_m,N);

	// Wait until work is done
	cudaDeviceSynchronize();

	testCUDA(cudaEventRecord(stop, 0));
	testCUDA(cudaEventSynchronize(stop));
	testCUDA(cudaEventElapsedTime(&timer, start, stop));

	// Copy result from GPU RAM into CPU RAM
	testCUDA(cudaMemcpy(M, dev_m, N*SIZE_M*sizeof(int), cudaMemcpyDeviceToHost));

	bool sorted = true;	
    for(int k=0 ; k<N ; ++k) {
	sorted = check_array_sorted_no_print(&M[k*SIZE_M], SIZE_M);
	if(!sorted) break;
    }
	if(sorted) {
		printf("Batch SORT : OK\n");
		printf("======================================================\n");
		printf("Time to merge %d arrays of size %d : %f ms\n",N,SIZE_M,timer);
		printf("======================================================\n");
	}
	else {
		printf("BATCH SORT : KO\n");
	}

	// Free both CPU and GPU memory allocated
	testCUDA(cudaFree(dev_m));
	testCUDA(cudaFree(dev_b));
	testCUDA(cudaFree(dev_a));
	free(M);
	free(B);
	free(A);
}


