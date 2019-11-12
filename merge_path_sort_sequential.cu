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

int main(int argc, char *argv[]) {
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