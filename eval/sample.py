import torch
import torch.nn.functional as F

# CUDA kernel for 1D convolution
cuda_kernel = """
extern "C" __global__ void conv1d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_width, int output_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < output_width && idy < out_channels * batch_size) {
        int b = idy / out_channels;
        int oc = idy % out_channels;
        
        float sum = bias[oc];
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < 3; ++k) {
                int iw = idx + k;
                if (iw < input_width) {
                    int input_idx = (b * in_channels + ic) * input_width + iw;
                    int weight_idx = (oc * in_channels + ic) * 3 + k;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        output[idy * output_width + idx] = sum;
    }
}
"""

# Compile the CUDA kernel
conv1d_cuda = torch.cuda.compile(cuda_kernel, name="conv1d_kernel")

# Custom Conv1d function using the CUDA kernel
class CustomConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        assert input.is_cuda and weight.is_cuda and bias.is_cuda
        
        batch_size, in_channels, input_width = input.shape
        out_channels, _, kernel_size = weight.shape
        output_width = input_width - kernel_size + 1
        
        output = torch.empty(batch_size, out_channels, output_width, device='cuda')
        
        threads_per_block = (16, 16)
        blocks = (
            (output_width + threads_per_block[0] - 1) // threads_per_block[0],
            (out_channels * batch_size + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        conv1d_cuda(
            grid=blocks,
            block=threads_per_block,
            args=[input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                  output.data_ptr(), batch_size, in_channels, out_channels,
                  input_width, output_width]
        )
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is not implemented for simplicity
        return None, None, None

# Function to use the custom Conv1d
def custom_conv1d(x, weight, bias):
    return CustomConv1d.apply(x, weight, bias)

# Test the custom Conv1d against F.conv1d
def test_conv1d():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    input_width = 10
    kernel_size = 3
    
    # Generate random input, weight, and bias
    x = torch.randn(batch_size, in_channels, input_width, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda')
    bias = torch.randn(out_channels, device='cuda')
    
    # Compute reference output using F.conv1d
    ref_output = F.conv1d(x, weight, bias)
    
    # Compute output using custom Conv1d
    custom_output = custom_conv1d(x, weight, bias)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(ref_output - custom_output))
    print(f"Maximum difference between outputs: {max_diff.item()}")
    
    if max_diff < 1e-5:
        print("Custom Conv1d implementation matches F.conv1d!")
    else:
        print("Custom Conv1d implementation does not match F.conv1d.")

# Run the test
test_conv1d()