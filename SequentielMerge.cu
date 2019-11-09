#include "cuda_header.h"

#define LENGTH_A 10
#define LENGTH_B 8

#define BLOCKSIZE 10


void sequential_merge_sort(int *A, int *B, int *M, int length_A, int length_B) {
	
    int i = 0, j = 0;
    int length_M = length_A + length_B;
    
    while (i + j < length_M) {
		if (j >= length_B || (i<length_A && A[i]<B[j])) {
            M[i + j] = A[i++];
        }
		else {
            M[i + j] = B[j++];
        }	
	}
}

void test() {
        int *A = generate_array(5);
        int *B = generate_array(6);
        int *M = (int*)malloc(11 * sizeof(int));
       
        sequential_merge_sort(A, B, M, 5, 6);
        print_array(M , 11, "M");
        sequential_merge_sort(B, A, M, 6, 5);
        print_array(M , 11, "M");

        free(A);
        free(B);
        free(M);
}

typedef struct {
        int x;
        int y;
} duo;

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
                        }else {
                                K.x = Q.x + 1;
                                K.y = Q.y + 1;
                        }
                }else {
                        P.x = Q.x - 1;
                        P.y = Q.y + 1;
                }
        }
}

void question_1() {
    
    // Allocate CPU buffers for three vectors (two input, one output).
    int* A = generate_array(LENGTH_A * sizeof(int));
    int* B = generate_array(LENGTH_B * sizeof(int));
    int* M = (int*)malloc((LENGTH_A + LENGTH_B) * sizeof(int));
   
    // Declare GPU buffers
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_m = 0;

    // Allocate GPU buffers for three vectors (two input, one output).
    testCUDA(cudaMalloc((void**)&dev_a, LENGTH_A * sizeof(int)));
    testCUDA(cudaMalloc((void**)&dev_b, LENGTH_B * sizeof(int)));
    testCUDA(cudaMalloc((void**)&dev_m, (LENGTH_A + LENGTH_B) * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    testCUDA(cudaMemcpy(dev_a, A, LENGTH_A * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(dev_b, B, LENGTH_B * sizeof(int), cudaMemcpyHostToDevice));

    // Launch merge kernel on GPU
    mergeSmall_k << <1, LENGTH_A+LENGTH_B >> > (dev_a, LENGTH_A, dev_b, LENGTH_B, dev_m);

    // Wait until work is done
    cudaDeviceSynchronize();

    // Copy result from GPU RAM into CPU RAM
    testCUDA(cudaMemcpy(M, dev_m, (LENGTH_A + LENGTH_B) * sizeof(int), cudaMemcpyDeviceToHost));

    
    print_array(A, LENGTH_A, "A");
    print_array(B, LENGTH_B, "B");
    print_array(M, LENGTH_A + LENGTH_B , "M");

    // Free both CPU and GPU memory allocated
    free(A);
    free(B);
    free(M);
    testCUDA(cudaFree(dev_a));
    testCUDA(cudaFree(dev_b));
    testCUDA(cudaFree(dev_m));
}

int main()
{   
    // initialize random seed
    srand(time(0));
    
    //test();
    question_1();

    return 0;
}