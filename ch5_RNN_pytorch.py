# RNN original paper 1993: 符號很古老，不建議啃，看別人教學比較好
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB

# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hn = self.rnn(x)
        return self.fc(hn.squeeze(0))

# Prepare data
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
train_data, test_data = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=10000)
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64, sort_within_batch=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Initialize model, loss, and optimizer
INPUT_DIM = len(TEXT.vocab)
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = SimpleRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, iterator, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in iterator:
            text, text_lengths = batch.text
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs.squeeze(1), batch.label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(iterator)}")

train(model, train_iterator, criterion, optimizer)
