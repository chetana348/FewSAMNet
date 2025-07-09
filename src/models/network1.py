from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers import SAMImageEncoder, SAMPromptEncoder, SAMMaskDecoder
from src.models.backbones.encoders import *
from src.models.segment_anything.build_sam import sam_model_registry
from src.models.segment_anything.modeling.common import LayerNorm2d, Reshaper
from src.config import Config
from src.utils.utils import preprocess

import torch
import torch.nn as nn

class MultiStreamSegmentor(nn.Module):
    def __init__(
        self,
        cnn_backbone: str = "resnet",
        sam_model = sam_model_registry[f'vit_b_{Config["img_size"]}'](None),
        use_pretrained: bool = False,
        bridge_dim: int = 256
    ):
        super().__init__()

        # Setup core image encoder from SAM
        self.sam_image_module = SAMImageEncoder(sam_model=sam_model, freeze=True)
        patch_dim = self.sam_image_module.embed_dim
        patch_grid = Config["img_size"] // self.sam_image_module.patch_size

        # Prompt encoder and mask decoder from SAM
        self.sam_prompt_module = SAMPromptEncoder(sam_model=sam_model, freeze=False)
        self.sam_mask_module = SAMMaskDecoder(sam_model=sam_model)

        # Optional CNN encoder (ResNet)
        if cnn_backbone == "resnet":
            self.aux_cnn_encoder = ResNet(depth=50, pretrained=use_pretrained)
            conv_feature_channels = self.aux_cnn_encoder.feature_dims
        elif cnn_backbone=='vgg':
            self.aux_cnn_encoder = VGG(pretrained=use_pretrained)
            conv_feature_channels = self.aux_cnn_encoder.feature_dims
            

        # Token selection indices from SAM
        self.token_sampling_points = self.sam_image_module.global_index

        # Align CNN encoder features to token dimensions
        self.feature_adapters = nn.ModuleList([
            Reshaper(Config["img_size"] // rescale, conv_feature_channels[i], 
                     patch_grid, patch_dim)
            for i, rescale in enumerate([4, 8, 16, 32])
        ])

        # Bridge convolutions to refine output features from encoder to decoder
        self.decoder_bridges = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(patch_dim, bridge_dim, kernel_size=2, stride=2),
                LayerNorm2d(bridge_dim),
                nn.GELU(),
                nn.ConvTranspose2d(bridge_dim, bridge_dim // 8, kernel_size=2, stride=2)
            )
            for _ in range(3)
        ])

        # Embedding upsampler for integrating multi-scale info
        self.embedding_decoder = nn.Sequential(
            nn.ConvTranspose2d(bridge_dim, bridge_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(bridge_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(bridge_dim // 4, bridge_dim // 8, kernel_size=2, stride=2),
        )
    
    @torch.no_grad()
    def infer(self, batch_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        # === Step 1: Normalize and batch input images ===
        stacked_images = torch.stack([
            preprocess(sample["image"], 
                       pixel_mean=Config["norm_mean"], 
                       pixel_std=Config["norm_std"]) 
            for sample in batch_inputs
        ], dim=0)
    
        # === Step 2: Extract CNN feature maps ===
        encoder_features = self.aux_cnn_encoder(stacked_images)  # List of 4 feature maps
        decoder_features = [None] * 3  # Placeholder for U-Net skip path
    
        # === Step 3: Generate patch tokens + positional encoding ===
        patch_tokens = self.sam_image_module._embed_with_positional(stacked_images)
    
        # === Step 4: Inject CNN features into transformer stream ===
        for stage in range(len(self.token_sampling_points)):
            for blk in range(2):
                patch_tokens = self.sam_image_module.forward_block(patch_tokens, stage * 3 + blk)
    
            adapted_feature = self.feature_adapters[stage](encoder_features[stage]).permute(0, 2, 3, 1)
            patch_tokens = patch_tokens + adapted_feature
            patch_tokens = self.sam_image_module.forward_block(patch_tokens, self.token_sampling_points[stage])
    
            # Save intermediate outputs for decoder bridges
            if stage < 3:
                skip_tensor = patch_tokens.permute(0, 3, 1, 2)
                decoder_features[2 - stage] = self.decoder_bridges[2 - stage](skip_tensor)
    
        # === Step 5: Get final transformer output ===
        encoded_image = self.sam_image_module._project_features(patch_tokens)
    
        # === Step 6: Combine multi-scale decoder features ===
        self.unet_fused_feature = self.embedding_decoder(encoded_image)
        for k in range(3):
            self.unet_fused_feature += decoder_features[k]
    
        # === Step 7: Perform inference per sample ===
        predictions = []
        for sample_input, image_token, skip_feat in zip(batch_inputs, encoded_image, self.unet_fused_feature):
            if sample_input.get("point_coords", None) is not None:
                prompt = (sample_input["point_coords"], sample_input["point_labels"])
            else:
                prompt = None
    
            sparse_embed, dense_embed = self.sam_prompt_module(
                points=prompt,
                boxes=sample_input.get("boxes", None),
                masks=sample_input.get("mask_inputs", None)
            )
    
            lowres_logits = self.sam_mask_module(
                img_embed=image_token.unsqueeze(0),
                pos_encoding=self.sam_prompt_module.positional_encoding(),
                sparse_embed=sparse_embed,
                dense_embed=dense_embed,
                skip_features=skip_feat.unsqueeze(0)
            )
    
            final_mask = self.restore_original_resolution(
                lowres_logits,
                network_input_size=sample_input["image"].shape[-2:],
                image_original_size=sample_input["original_size"]
            )
            binary_mask = final_mask > 0.0
    
            predictions.append({
                "masks": binary_mask,
                "low_res_logits": lowres_logits
            })
    
        return predictions

    def train_forward(self, sample_batch: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        # === Step 1: Preprocess input image tensors ===
        normalized_batch = torch.stack([
            preprocess(item["image"], pixel_mean=Config["norm_mean"], pixel_std=Config["norm_std"])
            for item in sample_batch
        ], dim=0)
    
        # === Step 2: Extract CNN features ===
        encoder_cnn_features = self.aux_cnn_encoder(normalized_batch)
        skip_connection_outputs = [None] * 3
    
        # === Step 3: Patch + position embedding ===
        patch_tokens = self.sam_image_module._embed_with_positional(normalized_batch)
    
        # === Step 4: Forward through SAM transformer with CNN fusion ===
        for level in range(len(self.token_sampling_points)):
            for block in range(2):
                patch_tokens = self.sam_image_module.forward_block(patch_tokens, level * 3 + block)
    
            adapted_feats = self.feature_adapters[level](encoder_cnn_features[level])
            patch_tokens += adapted_feats.permute(0, 2, 3, 1)
            patch_tokens = self.sam_image_module.forward_block(patch_tokens, self.token_sampling_points[level])
    
            if level < 3:
                tokens_shaped = patch_tokens.permute(0, 3, 1, 2)
                skip_connection_outputs[2 - level] = self.decoder_bridges[2 - level](tokens_shaped)
    
        # === Step 5: Final image representation ===
        final_patch_embed = self.sam_image_module._project_features(patch_tokens)
    
        # === Step 6: Fuse encoder-decoder features ===
        self.unet_fused_feature = self.embedding_decoder(final_patch_embed)
        for k in range(3):
            self.unet_fused_feature += skip_connection_outputs[k]
    
        # === Step 7: Generate predictions per sample ===
        forward_outputs = []
        for sample, visual_tokens, skip_tensor in zip(sample_batch, final_patch_embed, self.unet_fused_feature):
            if sample.get("point_coords", None) is not None:
                point_data = (sample["point_coords"], sample["point_labels"])
            else:
                point_data = None
    
            sparse_tok, dense_tok = self.sam_prompt_module(
                points=point_data,
                boxes=sample.get("boxes", None),
                masks=sample.get("mask_inputs", None)
            )
    
            decoder_output = self.sam_mask_module(
                img_embed=visual_tokens.unsqueeze(0),
                pos_encoding=self.sam_prompt_module.positional_encoding(),
                sparse_embed=sparse_tok,
                dense_embed=dense_tok,
                skip_features=skip_tensor.unsqueeze(0)
            )
    
            full_masks = self.resize_for_training(decoder_output)
    
            forward_outputs.append({
                "masks": full_masks,
                "low_res_logits": decoder_output
            })
    
        return forward_outputs

    def restore_original_resolution(
        self,
        pred_masks: torch.Tensor,
        network_input_size: Tuple[int, int],
        image_original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upsamples and trims predicted masks to match original image dimensions.
        """
        upsampled_to_input = F.interpolate(
            pred_masks,
            size=(self.sam_image_module.img_size, self.sam_image_module.img_size),
            mode="bilinear",
            align_corners=False
        )
    
        cropped = upsampled_to_input[..., :network_input_size[0], :network_input_size[1]]
    
        resized_to_original = F.interpolate(
            cropped,
            size=image_original_size,
            mode="bilinear",
            align_corners=False
        )
    
        return resized_to_original
    
    
    def resize_for_training(self, raw_masks: torch.Tensor) -> torch.Tensor:
        """
        Upsamples logits/masks to the training resolution expected for loss computation.
        """
        return F.interpolate(
            raw_masks,
            size=(self.sam_image_module.img_size, self.sam_image_module.img_size),
            mode="bilinear",
            align_corners=False
        )
    
    
    def forward(self, batch_input: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Main model dispatch method for both training and inference.
        """
        if self.training:
            return self.train_forward(batch_input)
        else:
            return self.infer(batch_input)
    
            
