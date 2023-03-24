"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder
from models.model.vis_encoder import VisEncoderLayer

class TransformerCaptcha(nn.Module):

    def __init__(self, trg_pad_idx, trg_sos_idx, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx= 0
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        # self.encoder = Encoder(d_model=d_model,
        #                        n_head=n_head,
        #                        max_len=max_len,
        #                        ffn_hidden=ffn_hidden,
        #                        enc_voc_size=enc_voc_size,
        #                        drop_prob=drop_prob,
        #                        n_layers=n_layers,
        #                        device=device)
        self.encoder = VisEncoderLayer(
            image_size=(60, 210),
            patch_size=30,
            d_model=d_model,
            n_head=n_head,
            ffn_hidden=ffn_hidden,
        )

        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

    def forward(self, src, trg):
        # src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)

        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * \
                   self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        print(k.shape)
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask
    

if __name__ == '__main__':
    tc = TransformerCaptcha(
        trg_pad_idx=1,
        trg_sos_idx=1,
        dec_voc_size=26,
        d_model=512,
        n_head=8,
        max_len=256,
        ffn_hidden=2048,
        n_layers=8,
        drop_prob=0.1,
        device='cpu'
    )
    d = tc(torch.randn(1, 3, 60, 210), torch.randn(1, 256, 64))
    print(d, d.shape)