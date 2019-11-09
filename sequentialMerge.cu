#include "cuda_header.h"

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

int main()
{
	int *A = generate_array(5);
	int *B = (int*)malloc(5 * sizeof(int));
	int *M = (int*)malloc(10 * sizeof(int));
	for (int i = 0; i < 5; ++i) {
		*(B + i) = (i * 5);
	}
	for (int i = 0; i < 5; ++i)
		printf("A[%d] = %d, ", i, *(A + i));
	printf("\n");
	for (int i = 0; i < 5; ++i)
		printf("B[%d] = %d, ", i, *(B + i));
	printf("\n");

	sequential_merge_sort(A, B, M, 5, 5);

	for (int i = 0; i < 10; ++i)
		printf("M[%d] = %d, ", i, *(M + i));

	free(A);
	free(B);
	free(M);
    return 0;
}