# Training Script for Atayal LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import AtayalLSTM
from preprocess import download_data, load_and_clean, build_vocab, encode_and_split

# 超参数
SEQ_LEN = 50
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.003
EMBED_DIM = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2


def train():
    # 准备数据
    download_data()
    clean_lines = load_and_clean()
    chars, char2idx, idx2char = build_vocab(clean_lines)
    inputs, targets = encode_and_split(clean_lines, char2idx, SEQ_LEN)

    # 转成 tensor
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    vocab_size = len(chars)
    model = AtayalLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_inputs)
            loss = criterion(
                output.view(-1, vocab_size),
                batch_targets.view(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()

