#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 8

namespace mxnet
{
namespace op
{

__constant__ float kernel[10000]; // assume kernel < 10000

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // assume C <= 12
    // use strategy 3: block size covers output, load only core part of input, access halo cells from global memory
    __shared__ float input_ds[TILE_WIDTH][12][TILE_WIDTH][TILE_WIDTH];

    int b = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int h = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z * TILE_WIDTH + threadIdx.z;

    int b_start = blockIdx.x * TILE_WIDTH;
    int h_start = blockIdx.y * TILE_WIDTH;
    int w_start = blockIdx.z * TILE_WIDTH;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define kernel4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    if (b < B && h + K/2 < H && w + K/2 < W) {
        for (int c = 0; c < C; ++c) {
            input_ds[b - b_start][c][h - h_start][w - w_start] = x4d(b, c, h + K/2, w + K/2);
        }
    }
    else {
        for (int c = 0; c < C; ++c) {
            input_ds[b - b_start][c][h - h_start][w - w_start] = 0.0;
        }
    }

    __syncthreads();

    if (b < B && h < H_out && w < W_out) {
        for (int m = 0; m < M; ++m) {
            float acc = 0.0;
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < K; ++p) {
                    for (int q = 0; q < K; ++q) {
                        if (h + p >= h_start + K/2 && h + p < h_start + K/2 + TILE_WIDTH 
                            && w + q >= w_start + K/2 && w + q < w_start + K/2 + TILE_WIDTH)
                            acc += input_ds[b - b_start][c][h+p-h_start-K/2][w+q-w_start-K/2] * kernel4d(m, c, p, q);
                        else
                            acc += x4d(b, c, h+p, w+q) * kernel4d(m, c, p, q);
                    }
                }
            }
            __syncthreads();
            y4d(b, m, h, w) = acc;
        }
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    printf("C = %d.\n", C);

    // copy kernel to constant memory
    if ((M * C * K * K) < 10000) {
        cudaMemcpyToSymbol(kernel, w.dptr_, sizeof(float) * M * C * K * K);
        printf("Successfully copy kernel to constant memory\n");
    }

    // Set the kernel dimensions
    dim3 gridDim(ceil(B * 1.0 / TILE_WIDTH), ceil(H * 1.0 / TILE_WIDTH), ceil(W * 1.0 / TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
