from src.dataset import SentimentDataset
from src.model import SentimentModel
from src.trainer import Trainer
from torch.utils.data import DataLoader
import torch

# Splitting the data into train and validation sets


print("Creating Datasets...")
train_dataset = SentimentDataset(
    "/Users/goncalopes/Documents/sntmnt_analysis/data/processed/train.txt"
)
val_dataset = SentimentDataset(
    "/Users/goncalopes/Documents/sntmnt_analysis/data/processed/val.txt"
)

print("Creating Dataloaders...")
train_dataloader = DataLoader(
    train_dataset, batch_size=64, collate_fn=train_dataset.collate_batch
)
val_dataloader = DataLoader(
    val_dataset, batch_size=64, collate_fn=val_dataset.collate_batch
)

print("Creating Model...")
vocab_size = len(train_dataset.vocab)
embed_dim = 100
hidden_dim = 128
output_dim = 2

model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)
model = model.to(torch.device("mps"))

print("Creating Trainer...")
trainer = Trainer(model, train_dataloader, val_dataloader, 10, 0.001)

trainer.train()
