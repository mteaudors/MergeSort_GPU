all: build

build: 
	nvcc -o merge_path_sort.exe merge_path_sort.cu

exec:
	.\merge_path_sort.exe
