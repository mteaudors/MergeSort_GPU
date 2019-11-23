#ifndef CUDA_HEADER_H
#define CUDA_HEADER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>

#define SIZE_A 16
#define SIZE_B 16
#define SIZE_M (SIZE_A + SIZE_B)
#define D 128

#define LENGTH_A (SIZE_A * sizeof(int))
#define LENGTH_B (SIZE_B * sizeof(int))
#define LENGTH_M (SIZE_M * sizeof(int))

#define ARRAY_NUMBER 64
#define ARRAY_SIZES 64
#define TWO_ARRAY_SIZE (ARRAY_SIZES * 2)
#define TOTAL_SIZE (ARRAY_NUMBER * ARRAY_SIZES)

#define ARRAY_LENGTH (ARRAY_SIZES * sizeof(int))
#define TOTAL_LENGTH (TOTAL_SIZE * sizeof(int))

#define BLOCKSIZE 64
#define GRIDSIZE(DATA_SIZE) (((DATA_SIZE) + (BLOCKSIZE) - 1)/(BLOCKSIZE))

typedef struct {
	int x;
	int y;
} duo;

void testcuda(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		fprintf(stderr, "error in file %s at line %d\n", file, line);
		fprintf(stderr, "error is %s : %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
		exit(1);
	}
}

#define testCUDA(error) (testcuda((error), __FILE__, __LINE__))

int* generate_array(int length) {
	int* arr = (int*) malloc(length*sizeof(int));
	int r = rand()%1000;
	for (int i = 0; i < length; ++i)
		arr[i] = i * (i+r);
	return arr;
}

int* generate_unsorted_array(int length) {
	int* arr = (int*) malloc(length*sizeof(int));
	for (int i = 0; i < length; ++i)
		arr[i] = rand()%10000;
	return arr;
}

void print_array(int *arr, int length, std::string name) {
	printf("\n\t\t############### %s ###############\n\n",name.c_str());
	for (int i = 0; i < length ; ++i) {
		printf("%s[%d] = %d", name.c_str(), i, *(arr + i));
		
		if(i>0 && arr[i-1] > arr[i]) {
			printf("\nArray not sorted : %s[%d] = %d > %s[%d] = %d\n",name.c_str(),i-1,arr[i-1],name.c_str(),i,arr[i]);
			break;
        }
		if(i != length-1) printf(", ");
	}
	printf("\n");
}

void print_unsorted_array(int *arr, int length, std::string name) {
	printf("\n############### %s ###############\n\n",name.c_str());
	for (int i = 0; i < length ; ++i) {
		printf("%s[%d] = %d", name.c_str(), i, *(arr + i));
		if(i != length-1) printf(", ");
	}
	printf("\n");
}

#endif // CUDA_HEADER_H