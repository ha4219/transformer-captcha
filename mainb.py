import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import AdamW
from tqdm import tqdm

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

class IAMDataset(Dataset):
    def __init__(self, paths, processor, max_target_length=128):
        self.paths = paths
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # get file name + text 
        fn = self.paths[idx]
        text = fn.split('/')[-1].split('.')[0]
        # prepare image (i.e. resize + normalize)
        image = Image.open(fn).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
import glob
from conf import *

p1 = glob.glob(path1)
p2 = glob.glob(path2)
p3 = glob.glob(path3)
min_length = min([len(p1), len(p2), len(p3)])
all_path = p1[:min_length] + p2[:min_length] + p3[:min_length]
all_path = p2
train_path, test_path = train_test_split(all_path, random_state=42)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")


train_dataset = IAMDataset(paths=train_path, processor=processor)
eval_dataset = IAMDataset(paths=test_path, processor=processor)


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


encoding = train_dataset[0]
for k,v in encoding.items():
    print(k, v.shape)

image = Image.open(train_path[0]).convert("RGB")

labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=16)


from transformers import VisionEncoderDecoderModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


from datasets import load_metric

cer_metric = load_metric("cer")

import datetime
fnn = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9),"JST")).strftime("%Y%m%dT%H%M%S")

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def train():
    

    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_loader_length = len(train_dataloader)
    eval_loader_length = len(eval_dataloader)
    best_loss = 1e9
    train_loss_list, eval_loss_list = [], []

    for epoch in range(epochs):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            # get the inputs
            for k,v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_description(f'{epoch}, train_loss: {train_loss / train_loader_length}')
        # print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
        train_loss_list.append(train_loss / train_loader_length)
        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            pbar = tqdm(eval_dataloader)
            for batch in pbar:
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer
                pbar.set_description(f'{epoch}, eval_loss: {valid_cer / eval_loader_length}')
        
        total_cer = valid_cer / len(eval_dataloader)
        eval_loss_list.append(total_cer)


        # print("Validation CER:", total_cer)

        if total_cer < best_loss:
            best_loss = total_cer
            save_pretrained_dir = f'save/{fnn}_best'
            model.save_pretrained(save_pretrained_dir)

    model.save_pretrained("save/inter")

    import matplotlib.pyplot as plt

    length = len(train_loss_list)
    plt.plot(range(length), train_loss_list, label='train_loss')
    plt.plot(range(length), eval_loss_list, label='eval_loss')
    plt.legend()
    plt.savefig(f'vis/inter/{fnn}_plot.png')

def test():
    import matplotlib.pyplot as plt

    pair = []

    model2 = VisionEncoderDecoderModel.from_pretrained("save/20230330T213553_best")
    total = 0
    ans = 0

    for file_name in tqdm(test_path):
        tmp_total = 0
        image = Image.open(file_name).convert("RGB")
        fn = file_name.split('/')[-1].split('.')[0]
        # display(image)
        

        generated_ids = model2.generate(processor(image, return_tensors="pt").pixel_values)
        name = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        plt.imshow(image)
        plt.axis('off')
        plt.title(name)

        for i in range(len(fn)):
            if i < len(name) and fn[i] == name[i]:
                tmp_total += 1
        
        for i in range(max(len(fn), len(name))):
            if i<len(fn) and i<len(name):
                pair.append((fn[i], name[i]))
            elif i>=len(fn):
                pair.append((' ', name[i]))
            else:
                pair.append((fn[i], ' '))
        if name == fn:
            ans += 1
        
        if tmp_total == len(fn):
            plt.savefig(f'vis/inter/suc/{fn}.png')
        else:
            plt.savefig(f'vis/inter/fai/{fn}.png')

        tmp_total /= len(fn)
        total += tmp_total
        
    import pickle
    with open("pair.npy", "wb") as f:
        pickle.dump(pair, f)

    print('acc:', total / len(test_path))
    print('ans:', ans / len(test_path))


if __name__ == '__main__':
    # train()
    test()