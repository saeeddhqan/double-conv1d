// Copyright (c) 2023 Dan Fu, Hermann Kumbong

// Simple 1D depthwise convolution implementation with dilation and stride = 1
#include "shared.h"

const uint BX = 256;
const uint BY = 1;
const uint BZ = 1;

const uint TILE_SIZE_L = 4;
const uint TILE_SIZE_D = 1;

template<typename T, typename U>
__forceinline__ __device__ T _conv1d_k_3(const T* u, const U* weights, const U* bias, uint padding, uint l, uint d, uint L, uint D, uint K)
{
    T tmp;
    T weight;

    set_value(&tmp, bias[d]);

    int idx = l - padding;

    if(idx >= 0 && idx < L){
        set_value(&weight, weights[0]);
        tmp = __hfma(u[d * L + idx], weight, tmp);
    }
    
    idx++;
    if(idx >= 0 && idx < L){
        set_value(&weight, weights[1]);
        tmp = __hfma(u[d * L + idx], weight, tmp);
    }

    idx++;
    if(idx >= 0 && idx < L){
        set_value(&weight, weights[2]);
        tmp = __hfma(u[d * L + idx], weight, tmp);
    }

    return tmp;
}

template<typename T, typename U>
__global__ void conv1d_kernel(
    const T *__restrict__ u,
    const U *__restrict__ weights,
    const U *__restrict__ bias,
    T *__restrict__ out,
    uint padding,
    uint B,
    uint L,
    uint D,
    uint K,
    uint L_out
    )
{

    // // int C_out = 1;
    // uint idx = blockIdx.x * ((L * D)) + blockIdx.y * L;
    // uint idy = blockIdx.y;
    // uint idw = blockIdx.y * L;
    // int L_outx = L - K + 1;
    // int i;
    // printf("%d\n", L_outx);
    // for (int seq = 0; seq < L_outx; ++seq) {
    //     T sum;
    //     T weight;
    //     set_value(&sum, bias[0]);
    //     int i = 0;
    //     set_value(&weight, weights[idw]);
    //     sum = __hfma(u[idx + seq], weight, sum);
    //     i++;
    //     set_value(&weight, weights[idw + i]);
    //     sum = __hfma(u[idx + seq + i], weight, sum);
    //     i++;
    //     set_value(&weight, weights[idw + i]);
    //     sum = __hfma(u[idx + seq + i], weight, sum);
    //     out[idx + seq] = sum;
    // }
}

torch::Tensor convs(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias)
{
    const uint b = x.size(0);
    const uint d = x.size(1);
    const uint l = x.size(2);


    const uint k = 3;

    uint l_out = (l - k + 1);

    dim3 grid(1, 1);
    dim3 block(16, 16);
    // dim3 block(2998);
    // dim3 blockDims(BX, BY, BZ);

    // dim3 gridDims(ceil(l_out * 1.0 / (BX * TILE_SIZE_L) ), ceil((d * 1.0) / (BY * TILE_SIZE_D)), ceil((b * 1.0) / BZ));

    torch::Tensor out = torch::empty({b, d, l_out}, x.options());

    DISPATCH_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), weight.scalar_type(),
        "conv1d",
        ([&]
            { conv1d_kernel<input_t, weight_t><<<grid, block>>>(
                    static_cast<input_t *>(x.data_ptr()),
                    static_cast<weight_t *>(weight.data_ptr()),
                    static_cast<weight_t *>(bias.data_ptr()),
                    static_cast<input_t *>(out.data_ptr()),
                    0,
                    b,
                    l,
                    d,
                    k,
                    l_out
                    ); 
            }
        )
    );
    return out;
}