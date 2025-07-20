import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry

# ==== Load PyTorch decoder ====
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.eval()
mask_decoder = sam.mask_decoder
prompt_encoder = sam.prompt_encoder

# ==== Load TensorRT encoder ====
TRT_LOGGER = trt.Logger()
with open("sam_vit_b_encoder.trt", "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# ==== Prepare input image ====
image_bgr = cv2.imread("assets/demo1.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image = cv2.resize(image_rgb, (1024, 1024)).astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))[None, ...]  # NCHW
image = np.ascontiguousarray(image)

# ==== Encode image ====
input_name = engine.get_tensor_name(0)
context.set_input_shape(input_name, image.shape)
d_input = cuda.mem_alloc(image.nbytes)
context.set_tensor_address(input_name, int(d_input))
cuda.memcpy_htod(d_input, image)

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

# Run encoder inference
stream = cuda.Stream()
context.execute_async_v3(stream.handle)
stream.synchronize()
for name, buf in outputs.items():
    cuda.memcpy_dtoh(buf["host"], buf["device"])
    print(f"✅ Output {name} shape: {buf['host'].shape}")

# ==== Prepare decoder inputs ====
image_embedding = torch.from_numpy(outputs["embeddings"]["host"]).cpu()
image_pe = sam.prompt_encoder.get_dense_pe().cpu()  # Use true PE from prompt_encoder
interm_embeddings = [
    torch.from_numpy(outputs["input.48"]["host"]).cpu(),
    torch.from_numpy(outputs["input.72"]["host"]).cpu(),
    torch.from_numpy(outputs["onnx::Transpose_4000"]["host"]).cpu()
]

# Dummy point prompt (center of image)
point_coords = torch.tensor([[[512, 512]]], dtype=torch.float32)
point_labels = torch.tensor([[1]], dtype=torch.int32)

# Encode prompt
sparse_embeddings, dense_embeddings = prompt_encoder(
    points=(point_coords, point_labels),
    boxes=None,
    masks=None
)

# ==== Decode ====
with torch.no_grad():
    masks, iou_pred= mask_decoder(
        image_embeddings=image_embedding,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        hq_token_only=False,
        interm_embeddings=interm_embeddings
    )

print(f"✅ Masks shape: {masks.shape}")
print(f"✅ IOU predictions shape: {iou_pred.shape}")

# ==== Visualize ====
all_masks = torch.any(masks[0] > 0.0, dim=0).cpu().numpy()
mask_resized = cv2.resize(all_masks.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0]))

overlay = image_rgb.copy()
overlay[mask_resized > 0.0] = (
    overlay[mask_resized > 0.0] * 0.3 + np.array([0, 255, 0]) * 0.7
)
overlay = overlay.astype(np.uint8)

plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.axis("off")
plt.title("All Masks Overlay")
plt.show()


