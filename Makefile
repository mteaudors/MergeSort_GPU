all: build

build: 
	nvcc -o merge_batch.exe merge_batch.cu 
#	nvcc -o mergeSort.exe mergeSort.cu 
#	nvcc -o merge_path_sort.exe merge_path_sort.cu
#	nvcc -o merge_path_sort_multiple.exe merge_path_sort_multiple.cu
	

exec:
	.\merge_batch.exe
