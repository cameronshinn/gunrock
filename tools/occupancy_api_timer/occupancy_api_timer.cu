// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * occupancy_api_timer.cu
 *
 * @brief Test program to check runtimes of CUDA occupancy API functions.
 */

#include <gunrock/oprtr/1D_oprtr/for.cuh>
#include <chrono>
#include <iostream>

using namespace std;

template <typename T>
void occupancyApiTimer(unsigned int *maxPotentialBlockSizeMicros, unsigned int *MaxActiveBlocksPerMultiprocessorMicros, T func)
{
    using namespace chrono;
    int blockSize;
    int minGridSize;
    int maxActiveBlocks;

    // Time cudaOccupancyMaxPotentialBlockSize()
    steady_clock::time_point beginMpbs = steady_clock::now();
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0);
    steady_clock::time_point endMpbs = steady_clock::now();
    *maxPotentialBlockSizeMicros = duration_cast<microseconds>(endMpbs - beginMpbs).count();

    // Time cudaOccupancyMaxActiveBlocksPerMultiprocessor()
    steady_clock::time_point beginMabpm = steady_clock::now();
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, 0);
    steady_clock::time_point endMabpm = steady_clock::now();
    *maxPotentialBlockSizeMicros = duration_cast<microseconds>(endMabpm - beginMabpm).count();
}

int main(void) {
    unsigned int mpbsMicros, mabpmMicros;
    auto dummyLambda = [] __host__ __device__ (int a) { return a; };
    occupancyApiTimer(&mpbsMicros, &mabpmMicros, gunrock::oprtr::For_Kernel<decltype(dummyLambda)>);
    cout << "cudaOccupancyMaxPotentialBlockSize(): " << mpbsMicros / 1000000.0 << " seconds" << endl;
    cout << "cudaOccupancyMaxActiveBlocksPerMultiprocessor(): " << mabpmMicros / 1000000.0 << " seconds" << endl;
}
