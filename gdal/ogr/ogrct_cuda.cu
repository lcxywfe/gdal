#include <cuda_runtime_api.h>
#include <bits/huge_val.h>

namespace {

__global__ void degree2radian_wrap_kernel(const int count, double* x, double* y,
                                      const double dfSourceWrapLong) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < count) {
        if (x[i] != HUGE_VAL && y[i] != HUGE_VAL) {
            if (x[i] < dfSourceWrapLong - 180.0)
                x[i] += 360.0;
            else if (x[i] > dfSourceWrapLong + 180)
                x[i] -= 360.0;
        }
        i += gridDim.x * blockDim.x;
    }
}

__global__ void degree2radian_kernel(const int count, double* x, double* y,
                                      const double dfSourceToRadians) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < count) {
        if (x[i] != HUGE_VAL) {
            x[i] *= dfSourceToRadians;
            y[i] *= dfSourceToRadians;
        }
        i += gridDim.x * blockDim.x;
    }
}

__global__ void radian2degree_kernel(const int count, double* x, double* y,
                                      const double dfTargetFromRadians) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < count) {
        if (x[i] != HUGE_VAL && y[i] != HUGE_VAL) {
            x[i] *= dfTargetFromRadians;
            y[i] *= dfTargetFromRadians;
        }
        i += gridDim.x * blockDim.x;
    }
}

__global__ void radian2degree_wrap_kernel(const int count, double* x, double* y,
                                      const double dfTargetWrapLong) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < count) {
        if (x[i] != HUGE_VAL && y[i] != HUGE_VAL) {
            if (x[i] < dfTargetWrapLong - 180.0)
                x[i] += 360.0;
            else if (x[i] > dfTargetWrapLong + 180)
                x[i] -= 360.0;
        }
        i += gridDim.x * blockDim.x;
    }
}

__global__ void check_with_invert_kernel(const int count, double* x, double* y,
                                            const double* x_ori, const double* y_ori,
                                            const double* x_tar, const double* y_tar,
                                            const double dfThreshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < count) {
        if (x[i] != HUGE_VAL && y[i] != HUGE_VAL &&
            (fabs(x_tar[i] - x_ori[i]) > dfThreshold ||
             fabs(y_tar[i] - y_ori[i]) > dfThreshold)) {
            x[i] = HUGE_VAL;
            y[i] = HUGE_VAL;
        }
        i += gridDim.x * blockDim.x;
    }
}

__global__ void error_info_kernel(const int count, double* x, double* y,
                                  int* success) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < count) {
        if (x[i] == HUGE_VAL || y[i] == HUGE_VAL)
            success[i] = 0;
        else
            success[i] = 1;

        i += gridDim.x * blockDim.x;
    }
}

}  // anonymous namespace

namespace cu {

void degree2radian_wrap(const int grid_size, const int block_size, const int count,
                    double* x, double* y, const double dfSourceWrapLong) {
    degree2radian_wrap_kernel<<<grid_size, block_size>>>(count, x, y,
                                                     dfSourceWrapLong);
}

void degree2radian(const int grid_size, const int block_size, const int count,
                    double* x, double* y, const double dfSourceToRadians) {
    degree2radian_kernel<<<grid_size, block_size>>>(count, x, y,
                                                     dfSourceToRadians);
}

void radian2degree(const int grid_size, const int block_size, const int count,
                   double* x, double* y, const double dfTargetFromRadians) {
    radian2degree_kernel<<<grid_size, block_size>>>(count, x, y,
                                                    dfTargetFromRadians);
}

void radian2degree_wrap(const int grid_size, const int block_size,
                        const int count, double* x, double* y,
                        const double dfTargetWrapLong) {
    radian2degree_wrap_kernel<<<grid_size, block_size>>>(count, x, y,
                                                         dfTargetWrapLong);
}

void check_with_invert(const int grid_size, const int block_size,
                       const int count, double* x, double* y,
                       const double* x_ori, const double* y_ori,
                       const double* x_tar, const double* y_tar,
                       const double dfThreshold) {
    check_with_invert_kernel<<<grid_size, block_size>>>(
            count, x, y, x_ori, y_ori, x_tar, y_tar, dfThreshold);
}

void error_info(const int grid_size, const int block_size, const int count,
                double* x, double* y, int* success) {
    error_info_kernel<<<grid_size, block_size>>>(count, x, y, success);
}

}  // namespace cu