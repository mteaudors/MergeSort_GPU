#include "cuda_header.h"

#define LENGTH_A 1024
#define LENGTH_B 1024

void sequential_merge_sort(int *A, int *B, int *M, int length_A, int length_B) {
        int i = 0, j = 0;
        int length_M = length_A + length_B;
        while (i + j < length_M) {
                if (j >= length_B || *(A + i) < *(B + j))
                        M[i + j] = A[i++];
                else
                        M[i + j] = B[j++];
        }
}

void test() {
        int *A = generate_array(5);
        int *B = generate_array(6);
        int *M = (int*)malloc(11 * sizeof(int));
        for (int i = 0; i < 5; ++i)
                printf("A[%d] = %d, ", i, *(A + i));
        printf("\n");
        for (int i = 0; i < 6; ++i)
                printf("B[%d] = %d, ", i, *(B + i));
        printf("\n");

        sequential_merge_sort(A, B, M, 5, 6);

        for (int i = 0; i < 11; ++i)
                printf("M[%d] = %d, ", i, *(M + i));

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
        int* A = generate_array(LENGTH_A * sizeof(int));
        printf("A = \n");
        print_array(A, LENGTH_A);
        int* B = generate_array(LENGTH_B * sizeof(int));
        printf("B = \n");
        print_array(B, LENGTH_B);
        int* M = (int*)malloc((LENGTH_A + LENGTH_B) * sizeof(int));
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

        mergeSmall_k << <4, 512 >> > (dev_a, LENGTH_A, dev_b, LENGTH_B, dev_m);
        cudaDeviceSynchronize();
        printf("M = \n");
        testCUDA(cudaMemcpy(M, dev_m, (LENGTH_A + LENGTH_B) * sizeof(int), cudaMemcpyDeviceToHost));

        print_array(M, LENGTH_A + LENGTH_B);

        free(A);
        free(B);
        free(M);
}

int main()
{
        //test();
        question_1();
    return 0;
}