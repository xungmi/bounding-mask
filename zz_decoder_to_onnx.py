import torch
from segment_anything import sam_model_registry
import torch.nn as nn

class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        image_embeddings,             # (1, 256, 64, 64)
        image_pe,                     # (1, 256, 64, 64)
        sparse_prompt_embeddings,    # (1, num_points, 256)
        dense_prompt_embeddings,     # (1, 256, 64, 64)
        multimask_output,            # scalar bool tensor
        hq_token_only,               # scalar bool tensor
        interm_embeddings            # (1, 256, 64, 64)
    ):
        masks, iou_preds = self.decoder(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings
        )
        return masks, iou_preds

# Load full SAM model
checkpoint = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
sam.eval()

# Extract decoder
decoder = DecoderWrapper(sam.mask_decoder)
decoder.eval()

# Dummy inputs
image_embeddings = torch.randn(1, 256, 64, 64)              # output từ encoder
image_pe = torch.randn(1, 256, 64, 64)                       # positional encoding
sparse_prompt_embeddings = torch.randn(1, 2, 256)           # điểm prompt
dense_prompt_embeddings = torch.randn(1, 256, 64, 64)       # mask prompt
multimask_output = torch.tensor(True)                       # bool
hq_token_only = torch.tensor(False)                         # bool
interm_embeddings = torch.randn(1, 256, 64, 64)             # feature trung gian từ encoder

# Export to ONNX
torch.onnx.export(
    decoder,
    (
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        hq_token_only,
        interm_embeddings
    ),
    "sam_vit_b_decoder.onnx",
    input_names=[
        "image_embeddings",
        "image_pe",
        "sparse_prompt_embeddings",
        "dense_prompt_embeddings",
        "multimask_output",
        "hq_token_only",
        "interm_embeddings"
    ],
    output_names=["masks", "iou_predictions"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes={
        "sparse_prompt_embeddings": {1: "num_points"},
        "masks": {1: "num_masks"}  # optional
    }
)

print("✅ Exported decoder to sam_vit_b_decoder.onnx")
