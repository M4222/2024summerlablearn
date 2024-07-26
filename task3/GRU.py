import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import random_split

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Data loading
train_iter, test_iter = IMDB()

# Build vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Text processing pipeline
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

# Label mapping
label_map = {'pos': 1, 'neg': 0}

# Label processing pipeline
def label_pipeline(label):
    return label_map[label]

# Collate function for DataLoader
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

# GRU Model definition
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(GRUModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets=None):
        embedded = self.embedding(text, offsets)
        gru_out, _ = self.gru(embedded.unsqueeze(1))  # GRU expects (batch_size, seq_len, input_size)
        gru_out = gru_out[:, -1, :]  # Take the last time step output
        return self.fc(gru_out)

# Constants and hyperparameters
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 64
LR = 0.5
EPOCHS = 10

# Prepare datasets and dataloaders
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize GRU model, optimizer, and criterion
model = GRUModel(len(vocab), EMBED_DIM, HIDDEN_DIM, 2).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training function with loss recording
def train_with_loss(dataloader, optimizer, criterion, epoch, train_losses):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 50
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        text, offsets = text.to(device), offsets.to(device)  # Move data to GPU
        label = label.to(device)  # Move label to GPU
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        train_losses.append(loss.item())  # Record the loss
        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f} | train loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count, loss.item()))
            total_acc, total_count = 0, 0

# Evaluation function
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            text, offsets = text.to(device), offsets.to(device)  # Move data to GPU
            label = label.to(device)  # Move label to GPU
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            loss = criterion(predicted_label, label)
            total_loss += loss.item() * label.size(0)
    return total_loss / total_count, total_acc / total_count

# Training loop with loss recording
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(1, EPOCHS + 1):
    train_with_loss(train_dataloader, optimizer, criterion, epoch, train_losses)
    print('-' * 59)
    print(f"End of epoch {epoch}, evaluating on test set...")
    test_loss,test_acc = evaluate(test_dataloader)
    test_losses.append(test_loss)
    print(f"Test Loss: {test_loss:.3f}")

    # Calculate accuracy per class and average accuracy
    class_correct = [0] * 2
    class_total = [0] * 2
    with torch.no_grad():
        for data in test_dataloader:
            labels, texts, offsets = data
            labels, texts, offsets = labels.to(device), texts.to(device), offsets.to(device)
            outputs = model(texts, offsets)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print(f'Accuracy of {"pos" if i == 1 else "neg"}: {100 * class_correct[i] / class_total[i]:.2f}%')

    overall_accuracy = sum(class_correct) / sum(class_total)
    print(f'Overall Accuracy: {100 * overall_accuracy:.2f}%')
    test_accuracies.append(overall_accuracy)
    print('-' * 59)


