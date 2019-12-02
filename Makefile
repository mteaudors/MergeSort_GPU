all: build

all: huffman merge_batch merge_sort merge_path_sort merge_path_sort_multiple

huffman:
	nvcc -o huffman.exe huffman.cu
merge_batch:
	nvcc -o merge_batch.exe merge_batch.cu 
merge_sort:
	nvcc -o mergeSort.exe mergeSort.cu 
merge_path_sort:
	nvcc -o merge_path_sort.exe merge_path_sort.cu
merge_path_sort_multiple:
	nvcc -o merge_path_sort_multiple.exe merge_path_sort_multiple.cu
	

exec:
	.\huffman.exe
