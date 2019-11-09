#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

void testcuda(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

#define testCUDA(error) (testcuda(error, __FILE__, __LINE__))

int* generate_array(int length) {
	int* arr = (int*)malloc(length * sizeof(int));
	for (int i = 0; i < length; ++i)
		*(arr + i) = i * (length + 4);
	return arr;
}

void print_array(int *arr, int length) {
	for (int i = 0; i < length - 1; ++i)
		printf("arr[%d] = %d, ", i, *(arr + i));

	printf("arr[%d] = %d\n", length - 1, *(arr + length - 1));
}

#endif // CUDA_HEADER_H