import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

TRT_ENGINE_PATH = "sam_vit_b_encoder.trt"
TRT_LOGGER = trt.Logger()

# Load engine
with open(TRT_ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Input: image
image = cv2.imread("assets/demo1.jpg")
image = cv2.resize(image, (1024, 1024)).astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))[None, ...]  # NCHW
image = np.ascontiguousarray(image)

# Set input tensor
input_name = engine.get_tensor_name(0)
context.set_input_shape(input_name, image.shape)

# Allocate memory for input
d_input = cuda.mem_alloc(image.nbytes)
context.set_tensor_address(input_name, int(d_input))
cuda.memcpy_htod(d_input, image)

# Allocate memory and set addresses for all output tensors
outputs = {}
output_bindings = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        d_output = cuda.mem_alloc(nbytes)
        context.set_tensor_address(name, int(d_output))
        outputs[name] = {
            "device": d_output,
            "host": np.empty(shape, dtype=dtype),
            "shape": shape
        }

# Run inference
stream = cuda.Stream()
context.execute_async_v3(stream.handle)
stream.synchronize()

# Copy outputs back
for name, buf in outputs.items():
    cuda.memcpy_dtoh(buf["host"], buf["device"])
    print(f"✅ Output {name} shape: {buf['host'].shape}")

