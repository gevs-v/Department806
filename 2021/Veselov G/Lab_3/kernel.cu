#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

#define DIM 1024

__global__ void reductionDouble(double* vect, double* vecOut, int size)
{
	__shared__ double block[DIM];
	unsigned int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int i = threadIdx.x;
	if (globalIndex < size)
		block[i] = vect[globalIndex];
	else
		block[i] = 0;

	__syncthreads();

	for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
	{
		if (i < j)
			block[i] += powf(block[i + j], 2);

		__syncthreads();
	}
	if (i == 0)
		vecOut[blockIdx.x] = block[0];
}

void generate_numbers(double* vect, int size)
{
	srand(time(NULL));
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < size; i++)
	{
		if (i % 2 == 0)
		{
			vect[i] = (double)1 / ((double)(i * 2) + double(1));
		}
		else
		{
			vect[i] = -(double)1 / ((double)(i * 2) + double(1));
		}
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto fillingTime = duration_cast<microseconds>(t2 - t1).count();
	cout << "Time needed to fill matrix with random number: " << (double)fillingTime / 1000 << " ms." << endl;
}

void sumCPUDouble(double* vect, double& sum, double& time, int size)
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < size; i++)
		sum += powf(vect[i], 2);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto timeCPU = duration_cast<microseconds>(t2 - t1).count();
	time = (double)timeCPU;// / 1000;
	cout << "Time needed to calculate the sum of Doubles by CPU: " << time << " us." << endl;
	cout << "Sum od Doubles calculated by CPU is: " << sum << " " << endl;
}

void show_vector(double* vect, int size)
{
	for (int i = 0; i < size; i++)
		cout << vect[i] << " ";
	cout << endl;
}
void sumGPUDouble(double* vector, double* vectorOutput, double& time, int vec_size)
{
	int numInputElements = vec_size;
	int numOutputElements;
	int threadsPerBlock = DIM;
	double* dev_vec;
	double* dev_vecOut;
	float dev_time;
	cudaEvent_t start, stop;
	cudaSetDevice(0);
	checkCudaErrors(cudaMalloc((double**)&dev_vec, vec_size * sizeof(double)));
	checkCudaErrors(cudaMalloc((double**)&dev_vecOut, vec_size * sizeof(double)));
	checkCudaErrors(cudaMemcpy(dev_vec, vector, vec_size * sizeof(double), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	do
	{
		numOutputElements = numInputElements / (threadsPerBlock);
		if (numInputElements % (threadsPerBlock))
			numOutputElements++;
		reductionDouble << < numOutputElements, threadsPerBlock >> > (dev_vec, dev_vecOut, numInputElements);
		numInputElements = numOutputElements;
		if (numOutputElements > 1)
			reductionDouble << < numOutputElements, threadsPerBlock >> > (dev_vecOut, dev_vec, numInputElements);

	} while (numOutputElements > 1);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto timeCPU = duration_cast<microseconds>(t2 - t1).count();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dev_time, start, stop);
	time = (double)timeCPU;

	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(vector, dev_vec, vec_size * sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vectorOutput, dev_vecOut, vec_size * sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(dev_vec));
	checkCudaErrors(cudaFree(dev_vecOut));
}

int main()
{
	int vec_size;
	while (true)
	{
		cout << "How big matrix do you want to create: ";
		cin >> vec_size;

		double* vector = new double[vec_size];
		double* vecOutput = new double[vec_size];
		double sumCPU = 0;
		double timeCPU, timeGPUDoubles;

		generate_numbers(vector, vec_size);
		sumCPUDouble(vector, sumCPU, timeCPU, vec_size);
		sumGPUDouble(vector, vecOutput, timeGPUDoubles, vec_size);

		cout << "Time needed to calculate the sum of Doubles by GPU: " << timeGPUDoubles << " us." << endl;
		cout << "Sum of Doubles calculated by GPU: " << setprecision(12) << vecOutput[0] << endl;

		cout << "Difference between sums of Double: " << setprecision(12) << vecOutput[0] - sumCPU << endl;

		if (sumCPU == vecOutput[0])
			cout << "Sum calculated by CPU is equal to sum calculated by GPU." << endl;


		cout << "-------------------------------------------" << endl;
		cout << "SpeedUP CPU to GPU: " << timeCPU / timeGPUDoubles << " times." << endl;

		delete[] vector;
		delete[] vecOutput;
	}

	return 0;
}