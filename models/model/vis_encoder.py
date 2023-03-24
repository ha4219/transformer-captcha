import torch
from torch import nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.blocks.encoder_layer import EncoderLayer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class VisEncoderLayer(nn.Module):
    def __init__(self, *, image_size, patch_size, d_model, n_head, ffn_hidden, pool = 'cls', channels = 3, drop_prob = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image d_modelensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_d_model = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_d_model),
            nn.Linear(patch_d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # need ?
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
        ])

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        return x
    

if __name__ == '__main__':
    encoder = VisEncoderLayer(image_size=(60, 210), patch_size=30, d_model=512, n_head=8, ffn_hidden=2048)
    t = encoder(torch.randn(1, 3, 60, 210))
    print(t, t.shape)