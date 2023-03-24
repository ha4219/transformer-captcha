import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import torch
from torchvision import transforms
# from utils.torch_utils import torch_distributed_zero_first

from utils.encode import load_encoder




def create_dataloader(
                        path,
                        image_size,
                        vocab_size,
                        batch_size=4,
                        augment=False,
                        workers=8,
                        rank=-1
                    ):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
    ])
    dataset = TCDataset(
        path,
        vocab_size=vocab_size,
        transform=transform if augment else False
    )
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    batch_size = min(batch_size, len(dataset))
    loader = DataLoader
    return loader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=nw,
                ), dataset

class TCDataset(Dataset):
    def __init__(self, paths, vocab_size, transform=None) -> None:
        super().__init__()
        self.paths = paths
        self.transform = transform
        self.vocab_size = vocab_size
        self.w2v, self.v2w = load_encoder()

    def __len__(self) -> int:
        self.filelength = len(self.paths)
        return self.filelength
    
    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        trg = torch.zeros((self.vocab_size, self.vocab_size))
        trg[0, self.w2v['sos']] = 1
        for i, c in enumerate(label):
            trg[i+1, self.w2v[c]] = 1
        trg[len(label) + 1, self.w2v['eos']] = 1
        trg[len(label) + 1:, 2] = 1

        # trg = [self.w2v['sos']] + [self.w2v[ch] for ch in label] + [self.w2v['eos']] + [2 for i in range(self.vocab_size-2-len(label))]
        # trg = torch.LongTensor(trg)
        # print(trg.shape)
        return img, trg, label


if __name__ == '__main__':
    from glob import glob
    import matplotlib.pyplot as plt
    from conf import *
    td = TCDataset(glob(path), vocab_size=126)
    img, trg, label = td.__getitem__(1)
    plt.imshow(img)
    plt.title(label)
    plt.show()
    print(trg, label)
    plt.savefig('vis/test0.png')
    img, trg, label = td.__getitem__(0)
    plt.imshow(img)
    plt.title(label)
    plt.show()
    print(trg, label)
    plt.savefig('vis/test1.png')
