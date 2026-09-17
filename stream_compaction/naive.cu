#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScanStep(int n, int offset, int *odata, const int *idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            odata[k] = (k >= offset) ? idata[k] + idata[k - offset] : idata[k];
        }

        __global__ void kernInclusiveToExclusive(int n, int *odata, const int *idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            odata[k] = (k == 0) ? 0 : idata[k - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, int blockSize) {
            int *dev_A, *dev_B;
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_A failed!");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_B failed!");

            cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_A failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            int passes = ilog2ceil(n);
            for (int d = 1; d <= passes; d++) {
                int offset = 1 << (d - 1);
                kernNaiveScanStep<<<fullBlocksPerGrid, blockSize>>>(n, offset, dev_B, dev_A);
                std::swap(dev_A, dev_B);
            }
            // after the loop, dev_A holds the inclusive scan result
            kernInclusiveToExclusive<<<fullBlocksPerGrid, blockSize>>>(n, dev_B, dev_A);
            std::swap(dev_A, dev_B);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_A);
            cudaFree(dev_B);
        }
    }
}
