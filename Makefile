all: build

build: 
	nvcc -o sequentialMerge.exe sequentialMerge.cu

exec:
	./sequentialMerge.exe
