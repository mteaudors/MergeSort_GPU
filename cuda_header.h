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

#define testCUDA(error) (testcuda(error, __file__, __line__))

int* generate_array(int length) {
	int* arr = (int*)malloc(length * sizeof(int));
	for (int i = 0; i < length; ++i)
		*(arr + i) = i * (i + 4);
	return arr;
}

#endif // CUDA_HEADER_H