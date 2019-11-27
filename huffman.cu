#include "cuda_header.h"

// Consider ASCII codes between 48 and 122
#define NB_CHAR 75
#define CHAR_FIRST 48

__device__ int A_diag[NB_CHAR];
__device__ int B_diag[NB_CHAR];

typedef struct elem {
    char letter;
    int nb_apparition;
} elem;

typedef struct node {
    int size;
    int total_nb_apparition;
    elem e[NB_CHAR];
    struct node *father;
    struct node *left_child;
    struct node *right_child;

} node;

void mergeSortGPU (elem *M , int length);


elem* generate_letter_distribution(int nb_elem) {
    elem *M = (elem*)malloc(nb_elem*sizeof(elem));

    for(int i=0 ; i<nb_elem ; ++i) {
        M[i].letter = CHAR_FIRST + i;
        M[i].nb_apparition = rand()%1000;
    }
    return M;
}

node* generate_huffman_nodes(int nb_elem) {
    int nb_nodes = 2*nb_elem-1;
    node *nodes = (node*)malloc(nb_nodes*sizeof(node));

    elem *M = generate_letter_distribution(nb_elem);

    mergeSortGPU(M , nb_elem);

    for(int i=0 ; i<nb_elem ; ++i) {
        nodes[i].size = 1;
        nodes[i].e[0] = M[i];
        nodes[i].total_nb_apparition = nodes[i].e[0].nb_apparition;
        nodes[i].father = nullptr;
        nodes[i].left_child = nullptr;
        nodes[i].right_child = nullptr;
    }
    for(int i=nb_elem ; i<nb_nodes ; ++i) {
        nodes[i].size = 0;
        nodes[i].total_nb_apparition = 0;
        nodes[i].father = nullptr;
        nodes[i].left_child = nullptr;
        nodes[i].right_child = nullptr;
    }

    free(M);

    return nodes;
} 

void print_letter_distribution(elem *M, int nb_elem) {
    for(int i=0 ; i<nb_elem ; ++i) {
        printf("%d :\tChar : %c\tApparition : %d\n",i,M[i].letter,M[i].nb_apparition);
    }
}

void print_sorted_letter_distribution(elem *M, int nb_elem) {
    for(int i=0 ; i<nb_elem ; ++i) {
        if(i>0 && M[i-1].nb_apparition > M[i].nb_apparition) {
			printf("\nArray not sorted : M[%d] = %d > M[%d] = %d\n",i-1,M[i-1].nb_apparition,i,M[i].nb_apparition);
			break;
        }
        printf("%d :\tChar : %c\tApparition : %d\n",i,M[i].letter,M[i].nb_apparition);
    }
}

void print_node(node *node) {
    printf("Address : %p\n",node);
    printf("Size = %d\n",node->size);
    printf("Total apparition = %d\n",node->total_nb_apparition);
    for(int k=0 ; k<node->size ; ++k) {
        printf("\tElement %d : \tChar : %c\tApparition : %d\n",k,node->e[k].letter, node->e[k].nb_apparition);
    }
    printf("\n");
    printf("Father address : %p\n",node->father);
    printf("Left child address : %p\n",node->left_child);
    printf("Right child address : %p\n",node->right_child);
}

void print_nodes(node *nodes, int nb_nodes) {
    for(int i=0 ; i<nb_nodes ; ++i) {
        printf("\n=============================================== Node %d ===============================================\n\n",i);
        print_node(&nodes[i]);
    }
    printf("\n");
}

void print_tree(node *root) {
    //TODO
}

/**
	Find the intersection between one diagonal and the merge path.
*/
__global__ void pathBig_k(elem *A, int length_A, elem *B, int length_B, int start_diag) {
		int nb_threads = gridDim.x * blockDim.x;
		int i = threadIdx.x + blockIdx.x * blockDim.x;
        int index_diag = (i+start_diag) * (length_A + length_B) / nb_threads;
        
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
                if (Q.y >= 0 && Q.x <= length_B && (Q.y == length_A || Q.x == 0 || A[Q.y].nb_apparition > B[Q.x - 1].nb_apparition)) {
                        if (Q.x == length_B || Q.y == 0 || A[Q.y - 1].nb_apparition <= B[Q.x].nb_apparition) {
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

__global__ void mergeBig_k(elem *A, int length_A, elem *B, int length_B, elem* M, int start_diag) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nb_threads = gridDim.x * blockDim.x;
	int length = (length_A + length_B) / nb_threads;
	int start_M = i*(length_A+length_B)/ nb_threads;

	for(int k=0 ; k<length ; ++k) {
        int i = start_diag + start_M + k;
		if (A_diag[i] < length_A && (B_diag[i] == length_B || A[A_diag[i]].nb_apparition <= B[B_diag[i]].nb_apparition)) {
            M[start_M + k] = A[A_diag[i]];
		} else {
			M[start_M + k] = B[B_diag[i]];
		}
	}
}


void mergeSortGPU (elem *M , int length) {
    elem *M_dev, *M_dev_copy;
	int mergeSize = 2;

    testCUDA(cudaMalloc((void**)&M_dev , length*sizeof(elem)));
    testCUDA(cudaMalloc((void**)&M_dev_copy , length*sizeof(elem)));
    testCUDA(cudaMemcpy(M_dev, M,length*sizeof(elem), cudaMemcpyHostToDevice));
    
    while(mergeSize <= pow(2,ceil(log2(length)))) {
        testCUDA(cudaMemcpy(M_dev_copy, M_dev, length * sizeof(elem), cudaMemcpyDeviceToDevice));
		for (int k = 0; k < ((length+mergeSize-1) / mergeSize); ++k) {
            if(k<(length/mergeSize))
                pathBig_k << <1, mergeSize >> > (M_dev + k * mergeSize, mergeSize / 2, M_dev + (2 * k + 1)*(mergeSize / 2), mergeSize / 2, mergeSize*k);
            else { 
                // k==length/mergeSize
                int mergeSizeLast = length%mergeSize;
                if(mergeSizeLast > mergeSize/2) {
                    pathBig_k << <1, mergeSizeLast >> > (M_dev + k * mergeSize, mergeSize / 2,M_dev + k * mergeSize + mergeSize/2, mergeSizeLast - (mergeSize/2), mergeSize*k);
                }
            }
        }
        
        testCUDA(cudaDeviceSynchronize());
        for(int k=0 ; k<((length+mergeSize-1)/mergeSize) ; ++k) {
            if(k<(length/mergeSize))
                mergeBig_k<<<1,mergeSize>>>(M_dev_copy+k*mergeSize, mergeSize/2, M_dev_copy+(2*k+1)*(mergeSize/2), mergeSize/2, M_dev+k*mergeSize, mergeSize*k);
            else { 
                // k==length/mergeSize
                int mergeSizeLast = length%mergeSize;
                if(mergeSizeLast > mergeSize/2) {
                    mergeBig_k << <1, mergeSizeLast >> > (M_dev_copy + k * mergeSize, mergeSize / 2,M_dev_copy + k * mergeSize + mergeSize/2, mergeSizeLast - (mergeSize/2),  M_dev+k*mergeSize, mergeSize*k);
                }
            }
        }
		testCUDA(cudaDeviceSynchronize());
        mergeSize *= 2;
    }
    
    testCUDA(cudaMemcpy(M, M_dev,length*sizeof(elem), cudaMemcpyDeviceToHost));
    testCUDA(cudaFree(M_dev));
    testCUDA(cudaFree(M_dev_copy));
}


void test_mergeGPU() {
    elem* M = generate_letter_distribution(NB_CHAR);
    print_letter_distribution(M,NB_CHAR);
    mergeSortGPU(M , NB_CHAR);
    printf("-------------------------------------------------------------------------------------------\n");
    print_sorted_letter_distribution(M,NB_CHAR);
    free(M);
}

node* huffman_tree(node *nodes, int nb_elem, int nb_nodes) {
    int current_nb_elem = nb_elem;
    int min[2] = {0,1};

    while(current_nb_elem < nb_nodes) {
        nodes[current_nb_elem].size = nodes[min[0]].size + nodes[min[1]].size;
        nodes[current_nb_elem].total_nb_apparition = nodes[min[0]].total_nb_apparition + nodes[min[1]].total_nb_apparition;
        memcpy(nodes[current_nb_elem].e , nodes[min[0]].e, nodes[min[0]].size*sizeof(elem));
        memcpy(nodes[current_nb_elem].e + nodes[min[0]].size , nodes[min[1]].e, nodes[min[1]].size*sizeof(elem));

        nodes[min[0]].father = &nodes[current_nb_elem];
        nodes[min[1]].father = &nodes[current_nb_elem];

        nodes[current_nb_elem].left_child = &nodes[min[0]];
        nodes[current_nb_elem].right_child = &nodes[min[1]];

        int new_position = current_nb_elem-1;
        while(nodes[current_nb_elem].total_nb_apparition < nodes[new_position].total_nb_apparition) --new_position;

        node n = nodes[current_nb_elem];
        for(int k=current_nb_elem ; k>new_position ; --k) nodes[k] = nodes[k-1];
        nodes[new_position+1] = n;
        
        ++current_nb_elem; 
        min[0]+=2;
        min[1]+=2;       
    }
    return &nodes[nb_nodes-1];
}


int main(int argc , char *argv[]) {

    testCUDA(cudaDeviceReset());

    // initialize random seed
    srand(time(0));

    //test_mergeGPU();

    int nb_elem = NB_CHAR, 
        nb_nodes = 2*nb_elem-1;
    node *nodes = generate_huffman_nodes(nb_elem);

    node *root = huffman_tree(nodes, nb_elem, nb_nodes);
    print_nodes(nodes, nb_nodes);

    free(nodes);
   
}