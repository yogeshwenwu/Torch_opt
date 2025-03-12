import torch
import torch.nn as nn
import torch.fx
import time


# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

class SimpleModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2 + 1
        else:
            return x * 3 - 1

x = torch.randn(1)

traced_model = torch.jit.trace(SimpleModel(), x) #try passing -x

scripted_model = torch.jit.script(SimpleModel())

fx_model = torch.fx.symbolic_trace(SimpleModel(), concrete_args={"x": x})

compiled_model = torch.compile(SimpleModel())

def time_model_execution(model, input_tensor):
    start_time = time.time()
    output = model(input_tensor) 
    end_time = time.time()  
    execution_time = end_time - start_time 

    print("Run1")
    print(execution_time)


# --
    start_time = time.time()
    output = model(input_tensor) 
    end_time = time.time()
    execution_time = end_time - start_time
    print("Run2")

    print(execution_time)



    return output, execution_time

original_model = SimpleModel()
original_output, original_time = time_model_execution(original_model, x)
print(f"Original Model Output: {original_output}, Time: {original_time:.6f} seconds")

traced_output, traced_time = time_model_execution(traced_model, x)
print(f"Traced Model Output: {traced_output}, Time: {traced_time:.6f} seconds")

scripted_output, scripted_time = time_model_execution(scripted_model, x)
print(f"Scripted Model Output: {scripted_output}, Time: {scripted_time:.6f} seconds")

fx_output, fx_time = time_model_execution(fx_model, x)
print(f"FX Symbolic Trace Output: {fx_output}, Time: {fx_time:.6f} seconds")

compiled_output, compiled_time = time_model_execution(compiled_model, x)
print(f"Compiled Model Output: {compiled_output}, Time: {compiled_time:.6f} seconds")


print("\nTorchScript Trace Graph:")
print(traced_model.code)
print(traced_model.graph)


print("\nTorchScript Script Graph:")
print(scripted_model.code)

print(scripted_model.graph)

print("\nFX Symbolic Trace Graph:")
print(fx_model.graph)

print("\nCompiled Model (torch.compile) Info:")
print(compiled_model)
 