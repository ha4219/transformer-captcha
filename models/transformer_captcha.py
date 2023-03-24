import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.embedding.transformer_embedding import TransformerEmbedding
from models.embedding.token_embeddings import TokenEmbedding


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class Model(nn.Module):
    def __init__(self, image_size, patch_size, d_model, n_head, n_layers, vocab_size, channels = 3, drop_prob = 0.) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image d_modelensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_d_model = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_d_model),
            nn.Linear(patch_d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.input_pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.output_pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        self.tok_emb = nn.Linear(vocab_size, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x, tgt):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        input_pos_embedding = self.input_pos_embedding[:, :n, :]
        x += input_pos_embedding
        x = self.transformer_encoder(x)

        tgt = self.tok_emb(tgt)
        output_pos_embedding = self.output_pos_embedding[:, :n, :]
        tgt += output_pos_embedding
        tgt = self.drop_out(tgt)
        x = self.transformer_decoder(tgt, x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    model = Model(
        image_size=(60, 210),
        patch_size=10,
        d_model=512,
        n_head=8,
        n_layers=6,
        vocab_size=126,
    )

    x = torch.rand(1, 3, 60, 210)
    tgt = torch.LongTensor([[i for i in range(126)]])

    out = model(x, tgt)

    print(out.shape)