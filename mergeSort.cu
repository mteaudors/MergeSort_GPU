
#include "cuda_header.h"

__device__ int A_diag[D];
__device__ int B_diag[D];

/**
	Find the intersection between one diagonal and the merge path.
*/
__global__ void pathBig_k(int *A, int length_A, int *B, int length_B, int start_diag) {
		int nb_threads = gridDim.x * blockDim.x;
		int i = threadIdx.x + blockIdx.x * blockDim.x;
        int index_diag = (i+start_diag) * (length_A + length_B) / nb_threads;

        //printf("Writting in A_diag at index : %d\n",index_diag);
        
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

__global__ void mergeBig_k(int *A, int length_A, int *B, int length_B, int* M, int start_diag) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i*(length_A+length_B)/ nb_threads;

    __shared__ int A_index[D];
    __shared__ int B_index[D];

    for(int k=0 ; k<length ; ++k) {
		A_index[start_diag + start_M + k] = A_diag[start_diag + start_M + k];
		B_index[start_diag + start_M + k] = B_diag[start_diag + start_M + k];
	}

	for(int k=0 ; k<length ; ++k) {
        int i = start_diag + start_M + k;
		if (A_index[i] < length_A && (B_index[i] == length_B || A[A_index[i]] <= B[B_index[i]])) {
            M[start_M + k] = A[A_index[i]];
		} else {
			M[start_M + k] = B[B_index[i]];
		}
	}
}

/*
tidx = indice du thread dans son groupe de tableaux à trier, genre 32 tableaux de 32 cases à trier dans un block
Qt = indice du groupe de travail du thread actuel
gbx = indice du thread pour accéder à la mémoire globale, indice d'un groupe de thread dans l'ensemble des threads pour pouvoir accéder à la RAM
*/

void mergeSortGPU (int *M , int length) {
    int *M_dev, *M_dev_copy;
    testCUDA(cudaMalloc((void**)&M_dev , D*sizeof(int)));
    testCUDA(cudaMalloc((void**)&M_dev_copy , D*sizeof(int)));
    testCUDA(cudaMemcpy(M_dev, M,D*sizeof(int), cudaMemcpyHostToDevice));
    
    int mergeSize = 2;
    while(mergeSize <= length) {
		testCUDA(cudaMemcpy(M_dev_copy, M_dev, D * sizeof(int), cudaMemcpyDeviceToDevice));
        for(int k=0 ; k<length/mergeSize ; ++k) {
            pathBig_k<<<1,mergeSize>>>(M_dev+k*mergeSize, mergeSize/2, M_dev+(2*k+1)*(mergeSize/2), mergeSize/2, mergeSize*k);
			/*
			on doit enlever celui là car le merge_i doit attendre la fin de path_i, mais là dans l'immédiat on a le fait 
			que merge_i+1 attend path_i+1 et merge_i, ce qui n'est pas du tout nécessaire, d'où le fait qu'on doit faire
			un kernel qui lance les deux autres, pour avoir la seule synchro nécessaire
			*/
			testCUDA(cudaDeviceSynchronize());
			mergeBig_k<<<1,mergeSize>>>(M_dev_copy+k*mergeSize, mergeSize/2, M_dev_copy+(2*k+1)*(mergeSize/2), mergeSize/2, M_dev+k*mergeSize, mergeSize*k);
        }
		testCUDA(cudaDeviceSynchronize());
        mergeSize *= 2;
    }
    
    testCUDA(cudaMemcpy(M, M_dev,D*sizeof(int), cudaMemcpyDeviceToHost));
    testCUDA(cudaFree(M_dev));
    testCUDA(cudaFree(M_dev_copy));

}


int main(int argc , char *argv[]) {
    // initialize random seed
    srand(time(0));

    int* M = generate_unsorted_array(D);
    print_unsorted_array(M , D , "M");

    mergeSortGPU(M,D);

    print_array(M,D,"M");
    free(M);
}





