#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16 || x.dtype() == torch::kFloat32, #x " must be float16 or bfloat16 or float32")
#define CHECK_SAME_TYPE(x, y) TORCH_CHECK(x.dtype() == y.dtype(), #x " and " #y " must have the same dtype")

#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x)


torch::Tensor convs(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
);



torch::Tensor convs_fwd(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);
    CHECK_SAME_TYPE(w, b);
    return convs(x, w, b);
}

