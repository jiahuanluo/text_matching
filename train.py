from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class PairDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.lines = open(self.file_path).readlines()

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.strip().split("\t")
        X = " ".join(line[:2])
        y = int(line[-1])
        return X, y

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':

    model = BertForSequenceClassification.from_pretrained('./output')
    model.train()

    # optimizer = AdamW(model.parameters(), lr=1e-5)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    tokenizer = AutoTokenizer.from_pretrained('./output')
    text_batch = ["1 2 3 4 11", "5 6 7 8 9 10"]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    train_file = "./data/gaiic_track3_round1_train_20210228.tsv"
    train_data = PairDataset(train_file)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    total_loss = 0
    for epoch in range(20):
        for i, (x, y) in enumerate(train_loader):
            x = tokenizer(list(x), return_tensors='pt', padding=True)
            labels = y.unsqueeze(0)
            outputs = model(x['input_ids'], attention_mask=x['attention_mask'], labels=labels)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch: {} | Iter: {} | Loss: {:.3f}".format(epoch, i, total_loss / 10))
                total_loss = 0
