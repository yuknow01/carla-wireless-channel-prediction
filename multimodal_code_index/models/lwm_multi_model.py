import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


ELEMENT_LENGTH = 64 # 16 -> 64
D_MODEL = 128  # 64 -> 128
MAX_LEN = 129
N_LAYERS = 12
N_HEADS = 8  # 12 -> 8
D_FF = D_MODEL * 4
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
DROPOUT = 0.1

# image patch params
IMG_SIZE = 224
PATCH_SIZE = 16
IMG_CHANS = 3

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-16) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (D_K ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS)
        self.linear = nn.Linear(N_HEADS * D_V, D_MODEL)
        self.norm = LayerNormalization(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        output =  context.transpose(1, 2).contiguous().view(batch_size, -1, N_HEADS * D_V)
        output  = self.linear(output)
        return residual + self.dropout(output)

class ProwiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = LayerNormalization(D_MODEL)

    def forward(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return self.norm(x + self.dropout(output))

class Channel_Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, channel_ids):
        seq_len = channel_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=channel_ids.device)
        pos = pos.unsqueeze(0).expand_as(channel_ids[:, :, 0])
        tok_emb = self.proj(channel_ids)
        embedding = tok_emb + self.pos_embed(pos)
        return self.norm(embedding)

class Image_Embedding(nn.Module):
    """
    image : (B, 3, 224, 224) -> (B, N_patches, D_model)
    """
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IMG_CHANS, d_model=D_MODEL):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_model = d_model

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.n_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Embedding(self.n_patches, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, img):
        # img : (B, 3, 224, 224)
        x = self.proj(img)  # (B, D_model, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, D_model)

        batch_size = x.size(0)
        pos = torch.arange(self.n_patches, dtype=torch.long, device=img.device)
        pos = pos.unsqueeze(0).expand(batch_size, self.n_patches)

        imag_embedding = x + self.pos_embed(pos)

        return self.norm(imag_embedding)


class ChannelEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.norm = LayerNormalization(D_MODEL)
        self.ffn = ProwiseFeedForwardNet()

    def forward(self, x):
        x = self.self_attn(x, x, x)
        x = self.norm(x)
        x = self.ffn(x)
        return x

class MultiModalEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.norm = LayerNormalization(D_MODEL)
        self.ffn = ProwiseFeedForwardNet()

    def forward(self, x):
        x = self.self_attn(x, x, x)
        x = self.norm(x)
        x = self.ffn(x)
        return x

class FusionEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = MultiHeadAttention()
        self.norm = LayerNormalization(D_MODEL)
        self.ffn = ProwiseFeedForwardNet()

    def forward(self, channel_tokens, image_tokens):
        x = self.cross_attn(channel_tokens, image_tokens, image_tokens)
        x = self.norm(x)
        x = self.ffn(x)
        return x

class multi_modal_lwm(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_embedding = Channel_Embedding(ELEMENT_LENGTH, D_MODEL, MAX_LEN)
        self.image_embedding = Image_Embedding(IMG_SIZE, PATCH_SIZE, IMG_CHANS, D_MODEL)

        self.channel_encoder_layers = nn.ModuleList([ChannelEncoderLayer() for _ in range(N_LAYERS)])
        self.fusion_encoder_layers = nn.ModuleList([FusionEncoderLayer() for _ in range(N_LAYERS)])

    @classmethod
    def from_pretrained(cls, ckpt_name='model_weights.pth', device='cuda', use_auth_token=None):
        model = cls().to(device)

        ckpt_path = ckpt_name
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Model loaded successfully from {ckpt_path} to {device}")

        return model

    def forward(self, channel_ids, images):
        channel_tokens = self.channel_embedding(channel_ids)  # (B, seq_len, D_model)
        if images.dim() == 5:
            B, T, C, H, W = images.size()
            images = images.reshape(B * T, C, H, W)
            image_tokens = self.image_embedding(images)  # (B*T, n_patches, D_model)
            image_tokens = image_tokens.view(B, T, -1, D_MODEL)  # (B, T, n_patches, D_model)
            image_tokens = image_tokens.mean(dim=1)  # (B, n_patches, D_MODEL)
        else:
            image_tokens = self.image_embedding(images)  # (B, n_patches, D_model)

        for i in range(N_LAYERS):
            channel_tokens = self.channel_encoder_layers[i](channel_tokens)

        for i in range(N_LAYERS):
            channel_tokens = self.fusion_encoder_layers[i](channel_tokens, image_tokens)

        return channel_tokens
