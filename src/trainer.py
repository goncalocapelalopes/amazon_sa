

import torch
import torch.nn as nn
import tqdm
class Trainer:
    
    def __init__(self, model, train_data_loader, val_data_loader, epochs, learning_rate):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_acc = 0

        for epoch in range(self.epochs):
            print("Epoch", epoch+1)
            self.model.train()  # Set the model to training mode
            total_loss = 0.0
            n_batches = 0
            for text, labels in tqdm.tqdm(self.train_data_loader):
                optimizer.zero_grad()
                outputs = self.model(text)
                loss = criterion(outputs, labels)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                n_batches += 1
            average_loss = total_loss / n_batches
            print(f'Epoch {epoch+1}/{self.epochs}, Average Loss: {average_loss:.4f}')

            # Validation step
            val_acc = self.evaluate()
            print(f'Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}')

            # Checkpointing
            if val_acc > best_val_acc:
                print(f'Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving checkpoint.')
                best_val_acc = val_acc
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_accuracy': best_val_acc
                }, filename="checkpoint_best.pth")
    
    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        total_accuracy = 0
        total = 0

        with torch.no_grad():  # Operations inside don't track history
            for text, labels in self.val_data_loader:
                outputs = self.model(text)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                total_accuracy += (predicted == labels).sum().item()

        return total_accuracy / total

    def save_checkpoint(self, state, filename="checkpoint.pth"):
        torch.save(state, filename)