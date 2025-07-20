import torch
from segment_anything import sam_model_registry
import onnx

# Load model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.eval().cpu()

# Dummy input (1 image, 3x1024x1024)
dummy_input = torch.randn(1, 3, 1024, 1024).cpu()

# Export ONNX
torch.onnx.export(
    sam.image_encoder,
    dummy_input,
    "sam_vit_b_encoder.onnx",
    input_names=["image"],
    output_names=["embeddings"],
    opset_version=13,
    do_constant_folding=True
)

print("✅ Exported encoder to sam_vit_b_encoder.onnx")
