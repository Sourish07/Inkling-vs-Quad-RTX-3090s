#include <torch/extension.h>

at::Tensor nvfp4_linear(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& scale, const at::Tensor& scale2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &nvfp4_linear, "Single-expert NVFP4 Marlin linear (CUDA)");
}
