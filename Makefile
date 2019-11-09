all: build

build: 
	nvcc -o SequentialMerge.exe SequentialMerge.cu

exec:
	./SequentialMerge.exe
