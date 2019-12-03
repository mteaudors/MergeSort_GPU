all: build

all: huffman merge_batch merge_sort merge_path_sort merge_path_sort_multiple merge_path_sort_parallel_small

huffman:
	nvcc -o huffman.exe huffman.cu
merge_batch:
	nvcc -o merge_batch.exe merge_batch.cu 
merge_sort:
	nvcc -o merge_sort.exe merge_sort.cu 
merge_path_sort:
	nvcc -o merge_path_sort.exe merge_path_sort.cu
merge_path_sort_multiple:
	nvcc -o merge_path_sort_multiple.exe merge_path_sort_multiple.cu
	
merge_path_sort_parallel_small:
	nvcc -o merge_path_sort_parallel_small.exe merge_path_sort_parallel_small.cu
	

exec:
	.\mergeSort.exe
