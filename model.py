import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 32 * 1 * 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total * 100

def train_model():
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    logging.info("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, pin_memory=True)
    logging.info(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total model parameters: {total_params:,}")

    Path("logs").mkdir(exist_ok=True)
    
    # Clear previous logs
    open('logs/training_metrics.json', 'w').close()

    num_epochs = 10
    best_accuracy = 0.0
    
    logging.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate batch accuracy
            accuracy = calculate_accuracy(outputs, labels)
            
            running_loss += loss.item()
            running_acc += accuracy
            batch_count += 1
            
            # Update progress bar with both loss and accuracy
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })
            
            if i % 10 == 9:
                avg_loss = running_loss / 10
                avg_acc = running_acc / 10
                # Save metrics to file
                with open('logs/training_metrics.json', 'a') as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'batch': i + 1,
                        'loss': avg_loss,
                        'training_accuracy': avg_acc
                    }, f)
                    f.write('\n')
                running_loss = 0.0
                running_acc = 0.0

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        logging.info(f'Epoch {epoch+1}:')
        logging.info(f'  Test Loss: {avg_test_loss:.4f}')
        logging.info(f'  Test Accuracy: {test_accuracy:.2f}%')
        
        # Save test metrics
        with open('logs/test_metrics.json', 'a') as f:
            json.dump({
                'epoch': epoch + 1,
                'test_loss': avg_test_loss,
                'test_accuracy': test_accuracy
            }, f)
            f.write('\n')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'mnist_cnn_best.pth')
            logging.info(f'New best model saved with accuracy: {test_accuracy:.2f}%')

    logging.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Generate predictions for 10 random test images
    logging.info("Generating predictions for random test images...")
    model.eval()
    random_indices = random.sample(range(len(test_dataset)), 10)
    results = []
    
    for idx in random_indices:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            prob = torch.nn.functional.softmax(output, dim=1)
            predicted = output.argmax().item()
            confidence = prob[0][predicted].item()
        
        results.append({
            'index': idx,
            'true_label': label,
            'predicted': predicted,
            'confidence': confidence
        })
    
    with open('logs/test_results.json', 'w') as f:
        json.dump(results, f)
    
    logging.info("Test results saved to logs/test_results.json")

if __name__ == '__main__':
    train_model()