import torch
from torch import nn

from typing import Tuple
from src.models.segment_anything.modeling.common import LayerNorm2d
from src.models.segment_anything.modeling.mask_decoder import MLP


class SAMImageEncoder(nn.Module):
    def __init__(self, sam_model, freeze=True):
        super().__init__()
        self.sam_img_encoder = sam_model.image_encoder

        # Extract and expose model attributes
        self.patch_size = self.sam_img_encoder.patch_size
        self.depth = self.sam_img_encoder.depth
        self.embed_dim = self.sam_img_encoder.embed_dim
        self.img_size = self.sam_img_encoder.img_size
        self.global_index = self.sam_img_encoder.global_index

        if freeze:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.sam_img_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, prompts=None):
        """
        Full forward pass with optional per-layer prompt tokens.
        """
        x = self._embed_with_positional(x)
        for idx in range(len(self.sam_img_encoder.blocks)):
            block = self.sam_img_encoder.blocks[idx]
            prompt = prompts[:, idx, :, :] if prompts is not None else None
            x = block(x, prompt_tokens=prompt)
        return self._project_features(x)

    def _embed_with_positional(self, x):
        """
        Applies patch embedding and adds positional encoding if available.
        """
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        return x

    def forward_block(self, x, block_idx):
        """
        Forward a single block by index.
        """
        return self.sam_img_encoder.blocks[block_idx](x)

    def _project_features(self, x):
        """
        Applies neck projection to final features.
        """
        return self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))


class SAMPromptEncoder(nn.Module):
    def __init__(self, sam_model, freeze=True):
        super().__init__()
        self.prompt_encoder = sam_model.prompt_encoder

        if freeze:
            self._lock_parameters()

    def _lock_parameters(self):
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        """
        Generate sparse and dense prompt embeddings.
        """
        sparse, dense = self.prompt_encoder(points, boxes, masks)
        return sparse, dense

    def positional_encoding(self):
        """
        Access dense positional encoding from the encoder.
        """
        return self.prompt_encoder.get_dense_pe()


class SAMMaskDecoder(nn.Module):
    def __init__(self, sam_model, transformer_dim: int = 256):
        super().__init__()
        
        # Keep original SAM mask decoder
        self.sam_mask_decoder = sam_model.mask_decoder
        
        # Learnable medical token and hypernetwork
        self.token_embed = nn.Embedding(1, transformer_dim)
        self.hyper_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        
        # Feature refinement layers
        self.mask_feat_adapter = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1)
        )

    def forward(
        self,
        img_embed: torch.Tensor,
        pos_encoding: torch.Tensor,
        sparse_embed: torch.Tensor,
        dense_embed: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self._generate_masks(
            img_embed, pos_encoding, sparse_embed, dense_embed, skip_features
        )

    def _generate_masks(
        self,
        image_tokens: torch.Tensor,
        image_pos: torch.Tensor,
        sparse_prompts: torch.Tensor,
        dense_prompts: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Prepare transformer tokens
        prompt_token = self.token_embed.weight  # [1, C]
        batch_size = sparse_prompts.shape[0]
        prompt_token = prompt_token.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, C]
        full_token_seq = torch.cat([prompt_token, sparse_prompts], dim=1)   # [B, N+1, C]

        # Prepare image embeddings for transformer
        expanded_image = torch.repeat_interleave(image_tokens, batch_size, dim=0) + dense_prompts
        expanded_pos = torch.repeat_interleave(image_pos, batch_size, dim=0)
        b1, c1, h1, w1 = expanded_image.shape

        # Transformer forward
        token_output, encoded = self.sam_mask_decoder.transformer(expanded_image, expanded_pos, full_token_seq)
        refined_token = token_output[:, 0, :]  # [B, C]

        # Upsample SAM decoder output
        reshaped = encoded.transpose(1, 2).view(b1, c1, h1, w1)
        upsampled_sam = self.sam_mask_decoder.output_upscaling(reshaped)  # [B, C', H, W]
        b2, c2, h2, w2 = upsampled_sam.shape

        # Merge with UNet multi-scale features and apply final convolution
        fused_features = self.mask_feat_adapter(upsampled_sam) + decoder_features.repeat(b2, 1, 1, 1)

        # Step 6: Hypernetwork-based mask generation
        hyper_kernel = self.hyper_mlp(refined_token).unsqueeze(1)  # [B, 1, C']
        masks_out = torch.matmul(hyper_kernel, fused_features.view(b2, c2, h2 * w2))  # [B, 1, H*W]
        masks_out = masks_out.view(b2, -1, h2, w2)  # [B, 1, H, W]

        return masks_out

