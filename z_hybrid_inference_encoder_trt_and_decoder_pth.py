import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

# Load decoder
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to("cuda")
predictor = SamPredictor(sam)

# Load image
image_bgr = cv2.imread("assets/demo1.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (1024, 1024))
predictor.set_image(image_resized)

# Encode image using TensorRT
TRT_ENGINE_PATH = "sam_vit_b_encoder.trt"
TRT_LOGGER = trt.Logger()

with open(TRT_ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Prepare input
image_input = image_resized.astype(np.float32) / 255.0
image_input = np.transpose(image_input, (2, 0, 1))[None, ...]
image_input = np.ascontiguousarray(image_input)

# Set input tensor
input_name = engine.get_tensor_name(0)
context.set_input_shape(input_name, image_input.shape)
d_input = cuda.mem_alloc(image_input.nbytes)
context.set_tensor_address(input_name, int(d_input))
cuda.memcpy_htod(d_input, image_input)

# Allocate memory for all outputs
outputs = {}
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
        }

# Run inference
stream = cuda.Stream()
context.execute_async_v3(stream.handle)
stream.synchronize()

# Copy back outputs
for name, buf in outputs.items():
    cuda.memcpy_dtoh(buf["host"], buf["device"])
    print(f"✅ Output {name} shape: {buf['host'].shape}")

# Use embeddings from output
image_embedding = outputs["embeddings"]["host"]
image_embedding_torch = torch.tensor(image_embedding).to("cuda")

# Set manually since we skipped encoder in predictor
predictor.features = image_embedding_torch

# Predict using a box (as an example)
box = np.array([250, 250, 770, 770])
masks, scores, logits = predictor.predict(box=box, multimask_output=True)

# Show result
for i, mask in enumerate(masks):
    overlay = image_resized.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)  # Red
    alpha = 0.6  # độ đậm mask: càng cao thì càng nổi

    # Tô màu trực tiếp vào vùng mask
    overlay[mask] = ((1 - alpha) * image_resized[mask] + alpha * color).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Mask {i+1} - Score: {scores[i]:.3f}")
    plt.axis("off")

plt.show()
