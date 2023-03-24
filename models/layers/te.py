import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    def forward(self, x, mem):
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x, mem)
        return x


model = Model()
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)


out = model(tgt, memory)

print(out.shape)