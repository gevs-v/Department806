
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define MAX_VALUE 256
#define BLOCKS 32
#define THREADS 128

__global__ void count(int*, int*, int);
__global__ void merge(int*);
void print(const char*, int*, int);

int main(int argc, const char** argv) {
	
	for (int size = 0; size < 33; size++) {

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		int n = pow(2, size);
		int* A;
		int* B;
		int* C;
		int* dA;
		int* dB;

		// malloc host memory
		A = (int*)malloc(sizeof(int) * n);
		B = (int*)malloc(sizeof(int) * MAX_VALUE);
		C = (int*)malloc(sizeof(int) * n);

		cudaEventRecord(start);

		// dump random values in range 0, MAX_VALUE in array A.
		for (int i = 0; i < n; ++i)
			A[i] = int(rand() % MAX_VALUE);

		// malloc device memory
		cudaMalloc((void**)&dA, sizeof(int) * n);
		cudaMalloc((void**)&dB, sizeof(int) * BLOCKS * MAX_VALUE);

		// copy the content of array A to dA on the device.
		cudaMemcpy(dA, A, sizeof(int) * n, cudaMemcpyHostToDevice);

		// call the count kernal.
		count << <BLOCKS, THREADS >> > (dA, dB, n);

		// merge count ararys using the merge kernal.
		merge << <BLOCKS, THREADS >> > (dB);

		// copy result of the summation of counted arrays to host.
		cudaMemcpy(B, dB, sizeof(int) * MAX_VALUE, cudaMemcpyDeviceToHost);

		// construct the sorted array.
		for (int i = 0, j = 0; i < MAX_VALUE; ++i)
			for (int k = 0; k < B[i]; ++k, ++j)
				C[j] = i;

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("For % d elements time elapsed in ms : % f\n", n, milliseconds);

		// print unsorted and sorted arrays.
		/*print("Original", A, n);
		print("Sorted", C, n);*/

		// free device memory.
		cudaFree(dA);
		cudaFree(dB);
	}

	return EXIT_SUCCESS;

}

__global__ void count(int* A, int* B, int n) {

	int b_id = blockIdx.x,
		b_num = gridDim.x,
		b_size,
		b_offset,
		t_id = threadIdx.x,
		t_num = blockDim.x,
		t_size,
		t_offset,
		offset;

	// initialize a shared memory array to store the count for each block.
	__shared__ int count[MAX_VALUE];

	// set intial values to zeros. Each thread sets its own share to zero.
	t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
	offset = t_id * t_size;
	for (int i = offset; i < offset + t_size && i < MAX_VALUE; ++i)
		count[i] = 0;

	// wait until all threads have completed the initialization process.
	__syncthreads();

	// accumulate the counts of each value. Each thread counts a certain portain
	// of the unsorted array.
	b_size = (b_num > n ? 1 : n / b_num);
	b_offset = b_id * b_size;

	t_size = (t_num > b_size ? 1 : b_size / t_num);

	offset = b_offset + t_id * t_size;
	for (int i = offset; i < offset + t_size && i < b_offset + b_size && i < n; ++i)
		atomicAdd(&count[A[i]], 1);

	// wait until all threads have completed the couting phase.
	__syncthreads();

	// copy the block count into global memory. Each thread copies its portioin to 
	// the global memory.
	t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
	t_offset = t_id * t_size;
	offset = b_id * MAX_VALUE + t_offset;

	if (offset + t_size <= (b_id + 1) * MAX_VALUE)
		memcpy(&B[offset], &count[t_offset], sizeof(int) * t_size);

}

__global__ void merge(int* B) {

	int b_id = blockIdx.x,
		b_num = gridDim.x,
		b_size,
		b_offset,
		t_id = threadIdx.x,
		t_num = blockDim.x,
		t_size,
		offset;

	// loop through and merge until all arrays are merged.
	for (int i = b_num, j = 2; i != 1; i /= 2, j *= 2) {

		// each block will operate on b_size values which equal, the number of 
		// count arrays * size of count arrays / number of blocks / 2. The final 2
		// represents the merge process.
		b_size = i * MAX_VALUE / b_num / 2;
		b_offset = (b_id / j) * (j * MAX_VALUE) + b_size * (b_id % j);

		t_size = (t_num > b_size ? 1 : b_size / t_num);

		// calculate the offset that each thread will start at and sum counts.
		offset = b_offset + t_id * t_size;
		for (int k = offset, l = offset + (MAX_VALUE * (j / 2));
			k < offset + t_size && k < b_offset + b_size; ++k, ++l)
			B[k] += B[l];

		// wait untill all arrays are merged for every step.
		__syncthreads();

	}

}

// print array.
void print(const char* name, int* array, int size) {

	printf("%s = [%d", name, array[0]);
	for (int i = 1; i < size; ++i) printf(", %d", array[i]);
	printf("]\n");

}
