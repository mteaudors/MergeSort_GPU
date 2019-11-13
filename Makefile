all: build

build: 
	nvcc -o mergeSort.exe mergeSort.cu 
#	nvcc -o merge_path_sort.exe merge_path_sort.cu
#	nvcc -o merge_path_sort_multiple.exe merge_path_sort_multiple.cu
	

exec:
	.\mergeSort.exe
