#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>


void testcuda(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

#define testCUDA(error) (testcuda(error, __FILE__, __LINE__))

int* generate_array(int length) {
	int* arr = (int*)malloc(length * sizeof(int));
	int r = rand()%1000;
	for (int i = 0; i < length; ++i)
		*(arr + i) = i * (i+r);
	return arr;
}

void print_array(int *arr, int length, std::string name) {
	for (int i = 0; i < length ; ++i) {
		printf("%s[%d] = %d", name.c_str(), i, *(arr + i));
		
		if(i>0 && arr[i-1] > arr[i]) {
                printf("\n!!! Array not sorted : %s[%d] = %d > %s[%d] = %d !!!\n",name.c_str(),i-1,arr[i-1],name.c_str(),i,arr[i]);
                break;
        }

		if(i != length-1) printf(", ");
	}
	printf("\n");
}

#endif // CUDA_HEADER_H