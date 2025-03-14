import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
input_tensor = torch.randn(1, 10)

# # Try different backends
backends = ['eager', 'inductor']#, 'cudagraphs', 'onnxrt', 'openxla', 'tvm']
for backend in backends:
    compiled_model = torch.compile(model, backend=backend)
    output = compiled_model(input_tensor)
    print(f"Backend: {backend}, Output: {output}")
# print(torch._dynamo.list_backends())