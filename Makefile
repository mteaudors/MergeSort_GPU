all: build

build: 
	nvcc -o sequentielMerge.exe sequentielMerge.cu

exec:
	./sequentielMerge.exe
