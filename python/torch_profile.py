import torch
from torch.profiler import profile, record_function, ProfilerActivity

from VQVAE_v2 import *

# Assuming your model and data are on GPU
model = EncoderFloat(1, 128).cuda()  # e.g., EncoderFloat + DecoderFloat
input_tensor = torch.randn(128, 1, 8, 8, 8).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_forward"):
        output = model(input_tensor)
        

prof.export_chrome_trace("trace.json")  # View in chrome://tracing