
#include <torch/extension.h>
#include "convs/convs.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("convs_forward", &convs_fwd, "conv1d forward (CUDA)");
}
