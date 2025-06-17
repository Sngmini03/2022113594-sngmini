### from google.colab import drive
###drive.mount('/content/drive')

### !pip install transformers scikit-learn

#  XLNet-base 감성 분석

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.optim import AdamW


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_imdb_csv(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'imdb_train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'imdb_test.csv'))
    return train_df['review'].tolist(), train_df['sentiment'].tolist(), test_df['review'].tolist(), test_df['sentiment'].tolist()


def train_xlnet_imdb(data_path, model_name='xlnet-base-cased', num_labels=2, epochs=5, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    X_train, y_train, X_test, y_test = load_imdb_csv(data_path)

    train_dataset = IMDBDataset(X_train, y_train, tokenizer)
    test_dataset = IMDBDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_test_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(test_acc)

        print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(classification_report(all_labels, all_preds))

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title("Accuracy Curve")

    plt.show()


if __name__ == '__main__':
    base_dir = '/content/drive/MyDrive/nlp_final/aclImdb'
    data_path = os.path.join(base_dir, 'data')
    train_xlnet_imdb(data_path)
