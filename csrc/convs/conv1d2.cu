// #include <cuda.h>
// #include <cuda_runtime.h>

// #include <ATen/cuda/CUDAContext.h>
// #include <torch/extension.h>
#include "shared.h"

template <typename input_t, typename weight_t, int kNWarpsPerBlock, int kNChunksPerSequence>
__global__ void scan(
    const weight_t* __restrict__ gates,
    const input_t* __restrict__ tokens,
    input_t* __restrict__ out,
    const int batch_stride,
    const int dim_stride,
    const int batch_stride_o
) {
    __shared__ weight_t warpLastToken[kNWarpsPerBlock];
    const uint lane_id = threadIdx.x & 31; // x % 32

    int odd = lane_id & 1;

    weight_t x = tokens[(blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x] * gates[(256 * blockIdx.z) + threadIdx.x];

    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        weight_t prev_x = __shfl_up_sync(0xffffffff, x, delta);
        if (lane_id >= delta && odd) {
            x += prev_x;
            odd = (lane_id % (delta * 4)) == (delta * 4) - 1;
        }
    }
    __syncwarp();
    if (lane_id == 31) {
        warpLastToken[(threadIdx.x / 32)] = x;
    }
    __syncthreads();
    if (threadIdx.x % 256 == 255) {
        x += warpLastToken[0];
        x += warpLastToken[1];
        x += warpLastToken[2];
        x += warpLastToken[3];
        x += warpLastToken[4];
        x += warpLastToken[5];
        x += warpLastToken[6];
        out[(blockIdx.x * batch_stride_o) + (blockIdx.y * 64) + blockIdx.z] = x;
    }
}

// template <typename weight_t, typename torch_weight_t>
// void
// warpscan(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse) {
//     const auto strides = tokens.strides();
//     const int batch_stride = strides[0];
//     const int dim_stride = strides[1];
//     const int gate_stride = gates.size(0) / 8;
//     TORCH_CHECK(tokens.stride(-1) == 1 || tokens.size(-1) == 1);
//     TORCH_CHECK(gates.stride(-1) == 1 || gates.size(-1) == 1);
//     const int batch_stride_o = out.strides()[0];
//     const auto sizes = tokens.sizes();
//     const int batch_size = sizes[0];
//     const int dim = sizes[1];
//     const int seqlen = sizes[2];

//     auto stream = at::cuda::getCurrentCUDAStream().stream();
//     // I guess the number of blocks are high, the performance is poor
//     dim3 grid(batch_size, dim, out.strides()[1]); // 4 is sequential steps;

//     if (seqlen == 256) {
//         constexpr int kNWarpsPerBlock = 8;
//         constexpr int kNChunksPerSequence = 1;
//         scan<weight_t, kNWarpsPerBlock, kNChunksPerSequence><<<grid, seqlen>>>(
//             reinterpret_cast<weight_t*>(gates.data_ptr<torch_weight_t>()), reinterpret_cast<weight_t*>(tokens.data_ptr<torch_weight_t>()), reinterpret_cast<weight_t*>(out.data_ptr<torch_weight_t>()),
//             batch_stride, dim_stride, batch_stride_o
//         );
//     } else {
//         TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
//     }
// }

torch::Tensor warpscan(const torch::Tensor gates, const torch::Tensor tokens) {
    const auto strides = tokens.strides();
    const int batch_stride = strides[0];
    const int dim_stride = strides[1];

    const int B = tokens.size(0);
    const int D = tokens.size(1);
    const int seqlen = tokens.size(2);
    
    torch::Tensor out = torch::empty({B, D, gates.size(0)}, tokens.options());
    const int batch_stride_o = out.strides()[0];

    dim3 grid(B, D, out.strides()[1]); // 4 is sequential steps;
    // dim3 grid(1, 1, 1); // 4 is sequential steps;

    // DISPATCH_FLOAT_AND_HALF_AND_BF16(tokens.scalar_type(), gates.scalar_type(),
    //     "depthwise conv 1d fwd bhl",
    //     ([&]
    //         { scan<input_t, weight_t, 8, 1><<<grid, seqlen>>>(
    //                 static_cast<weight_t *>(gates.data_ptr()),
    //                 static_cast<input_t *>(tokens.data_ptr()),
    //                 static_cast<input_t *>(out.data_ptr()),
    //                 batch_stride, dim_stride, batch_stride_o
    //             );
    //         }
    //     )
    // ); 
    scan<__half, __half, 8, 1><<<grid, seqlen>>>(
        static_cast<__half *>(gates.data_ptr()),
        static_cast<__half *>(tokens.data_ptr()),
        static_cast<__half *>(out.data_ptr()),
        batch_stride, dim_stride, batch_stride_o
    );
    return out;
}
