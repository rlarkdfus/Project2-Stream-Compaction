#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweepOptimized(int n, int stride, int *data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            int idx = (k + 1) * stride - 1;
            data[idx] += data[idx - (stride >> 1)];
        }

        __global__ void kernDownSweepOptimized(int n, int stride, int *data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            int idx = (k + 1) * stride - 1;
            int left = idx - (stride >> 1);
            int t = data[left];
            data[left] = data[idx];
            data[idx] += t;
        }

        static void scanOnDeviceOptimized(int n, int *dev_data, int blockSize) {
            int levels = ilog2(n);

            for (int d = 0; d < levels; d++) {
                int stride = 1 << (d + 1);
                int numActive = n / stride;
                dim3 fullBlocksPerGrid((numActive + blockSize - 1) / blockSize);
                kernUpSweepOptimized<<<fullBlocksPerGrid, blockSize>>>(numActive, stride, dev_data);
            }

            cudaMemset(dev_data + (n - 1), 0, sizeof(int));

            for (int d = levels - 1; d >= 0; d--) {
                int stride = 1 << (d + 1);
                int numActive = n / stride;
                dim3 fullBlocksPerGrid((numActive + blockSize - 1) / blockSize);
                kernDownSweepOptimized<<<fullBlocksPerGrid, blockSize>>>(numActive, stride, dev_data);
            }
        }

        __global__ void kernUpSweep(int n, int stride, int *data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if ((k + 1) % stride == 0) {
                data[k] += data[k - (stride >> 1)];
            }
        }

        __global__ void kernDownSweep(int n, int stride, int *data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if ((k + 1) % stride == 0) {
                int left = k - (stride >> 1);
                int t = data[left];
                data[left] = data[k];
                data[k] += t;
            }
        }

        static void scanOnDevice(int n, int *dev_data, int blockSize) {
            int levels = ilog2(n);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            for (int d = 0; d < levels; d++) {
                int stride = 1 << (d + 1);
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(n, stride, dev_data);
            }

            cudaMemset(dev_data + (n - 1), 0, sizeof(int));

            for (int d = levels - 1; d >= 0; d--) {
                int stride = 1 << (d + 1);
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n, stride, dev_data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, int blockSize) {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed!");

            timer().startGpuTimer();

            scanOnDeviceOptimized(paddedN, dev_data, blockSize);
            // scanOnDevice(paddedN, dev_data, blockSize);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            const int blockSize = 128;
            int paddedN = 1 << ilog2ceil(n);

            int *dev_idata, *dev_odata, *dev_bools, *dev_scan;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_scan, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_scan failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");
            cudaMemset(dev_scan, 0, paddedN * sizeof(int));

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
            cudaMemcpy(dev_scan, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanOnDeviceOptimized(paddedN, dev_scan, blockSize);
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_scan);

            timer().endGpuTimer();

            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_scan + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_scan);

            return count;
        }
    }
}
