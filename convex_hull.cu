#include "cuda_header.h"
#include <algorithm>
#include <vector>

#define NB_POINTS 16

__device__ int A_diag[NB_POINTS];
__device__ int B_diag[NB_POINTS];

struct Point {
	int x, y;

	//Just to keep the comparison nearby
	bool operator <(const Point &p) const {
		return x < p.x || (x == p.x && y < p.y);
	}
};

int cross(const Point &O, const Point &A, const Point &B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

Point* generate_hull(int length) {
	Point* P = (Point*)malloc(length * sizeof(Point));
	for (int i = 0; i < length; ++i) {
		P[i].x = rand() % length;
		P[i].y = rand() % length;
	}
	return P;
}

bool contains(Point* P, int length, int x, int y) {
	for (int i = 0; i < length; ++i) {
		if (P[i].x == x && P[i].y == y)
			return true;
	}
	return false;
}

void print_number(int i) {
	if ((i + 1) < 10) {
		printf("0%d ", i + 1);
	}
	else {
		printf("%d ", i + 1);
	}
}

void print_points(Point* P, int size, int P_length) {
	printf("    ");
	for (int i = 0; i < size; ++i) {
		print_number(i);
	}
	printf("\n---");
	for (int i = 0; i < size; ++i) {
		printf("---");
	}
	printf("\n");
	for (int i = 0; i < size; ++i) {
		print_number(i);
		printf("|");
		for (int j = 0; j < size; ++j) {
			if (contains(P, P_length, i, j)) {
				printf("+  ");
			}
			else {
				printf("   ");
			}
		}
		printf("\n");
	}
}

/**
	Find the intersection between one diagonal and the merge path.
*/
__global__ void pathBig_k(Point *A, int length_A, Point *B, int length_B, int start_diag) {
	int nb_threads = gridDim.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int index_diag = (i + start_diag) * (length_A + length_B) / nb_threads;

	duo K, P, Q;
	if (i > length_A) {
		K.x = i - length_A;
		K.y = length_A;
		P.x = length_A;
		P.y = i - length_A;
	}
	else {
		K.x = 0;
		K.y = i;
		P.x = i;
		P.y = 0;
	}
	while (true) {
		int offset = abs(K.y - P.y) / 2;
		Q.x = K.x + offset;
		Q.y = K.y - offset;
		//faire gaffe ici
		if (Q.y >= 0 && Q.x <= length_B && (Q.y == length_A || Q.x == 0 || (B[Q.x - 1].x < A[Q.y].x || (B[Q.x - 1].x == A[Q.y].x && B[Q.x - 1].y < A[Q.y].y)))) {
			//ici aussi
			if (Q.x == length_B || Q.y == 0 || (A[Q.y - 1].x < B[Q.x].x || (A[Q.y - 1].x == B[Q.x].x && A[Q.y - 1].y < B[Q.x].y))) {
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

__global__ void mergeBig_k(Point *A, int length_A, Point *B, int length_B, Point* M, int start_diag) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i * (length_A + length_B) / nb_threads;

	for (int k = 0; k < length; ++k) {
		int i = start_diag + start_M + k;
		//et ici
		if (A_diag[i] < length_A && (B_diag[i] == length_B || (A[A_diag[i]].x < B[B_diag[i]].x || (A[A_diag[i]].x == B[B_diag[i]].x && A[A_diag[i]].y < B[B_diag[i]].y)))) {
			M[start_M + k] = A[A_diag[i]];
		}
		else {
			M[start_M + k] = B[B_diag[i]];
		}
	}
}


void mergeSortGPU(Point *M, int length) {
	Point *M_dev, *M_dev_copy;
	int mergeSize = 2;

	testCUDA(cudaMalloc((void**)&M_dev, length * sizeof(Point)));
	testCUDA(cudaMalloc((void**)&M_dev_copy, length * sizeof(Point)));
	testCUDA(cudaMemcpy(M_dev, M, length * sizeof(Point), cudaMemcpyHostToDevice));

	while (mergeSize <= pow(2, ceil(log2(length)))) {
		testCUDA(cudaMemcpy(M_dev_copy, M_dev, length * sizeof(Point), cudaMemcpyDeviceToDevice));
		for (int k = 0; k < ((length + mergeSize - 1) / mergeSize); ++k) {
			if (k < (length / mergeSize))
				pathBig_k << <1, mergeSize >> > (M_dev + k * mergeSize, mergeSize / 2, M_dev + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, mergeSize*k);
			else {
				// k==length/mergeSize
				int mergeSizeLast = length % mergeSize;
				if (mergeSizeLast > mergeSize / 2) {
					pathBig_k << <1, mergeSizeLast >> > (M_dev + k * mergeSize, mergeSize / 2, M_dev + k * mergeSize + mergeSize / 2, mergeSizeLast - (mergeSize / 2), mergeSize*k);
				}
			}
		}

		testCUDA(cudaDeviceSynchronize());
		for (int k = 0; k < ((length + mergeSize - 1) / mergeSize); ++k) {
			if (k < (length / mergeSize))
				mergeBig_k << <1, mergeSize >> > (M_dev_copy + k * mergeSize, mergeSize / 2, M_dev_copy + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, M_dev + k * mergeSize, mergeSize*k);
			else {
				// k==length/mergeSize
				int mergeSizeLast = length % mergeSize;
				if (mergeSizeLast > mergeSize / 2) {
					mergeBig_k << <1, mergeSizeLast >> > (M_dev_copy + k * mergeSize, mergeSize / 2, M_dev_copy + k * mergeSize + mergeSize / 2, mergeSizeLast - (mergeSize / 2), M_dev + k * mergeSize, mergeSize*k);
				}
			}
		}
		testCUDA(cudaDeviceSynchronize());
		mergeSize *= 2;
	}

	testCUDA(cudaMemcpy(M, M_dev, length * sizeof(Point), cudaMemcpyDeviceToHost));
	testCUDA(cudaFree(M_dev));
	testCUDA(cudaFree(M_dev_copy));
}

Point* convex_hull(Point* P, int n, int *h_length)
{
	int k = 0;
	Point* H = (Point*) malloc(n * sizeof(Point));

	// Sort points lexicographically
	mergeSortGPU(P, n);

	// Build lower hull
	for (int i = 0; i < n; ++i) {
		while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	// Build upper hull
	for (int i = n - 1, t = k + 1; i > 0; --i) {
		while (k >= t && cross(H[k - 2], H[k - 1], P[i - 1]) <= 0) k--;
		H[k++] = P[i - 1];
	}
	*h_length = k;
	return H;
}


int main(int argc, char *argv[]) {

	// initialize random seed
	srand(time(0));

	for (int i = 0; i < 16; ++i) {
		testCUDA(cudaDeviceReset());
		int h_length = 0;


		Point* P = generate_hull(NB_POINTS);
		printf("The set of points we want to get the convex hull of is : \n");
		print_points(P, NB_POINTS, NB_POINTS);

		Point* H = convex_hull(P, NB_POINTS, &h_length);
		printf("The convex hull associated is : \n");
		print_points(H, NB_POINTS, h_length);

		free(H);
		free(P);
	}
}