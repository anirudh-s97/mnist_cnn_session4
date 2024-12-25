import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import random
import numpy as np
from model import CNN
import os
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check if CUDA is available and log device info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Dataset sizes: Training={len(train_dataset)}, Test={len(test_dataset)}")
print(f"Number of batches: Training={len(train_loader)}, Test={len(test_loader)}")

# Initialize model, loss, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def evaluate_model(data_loader, desc="Evaluating"):
    """Helper function to evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_sum += loss.item()
            
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            for i in range(len(target)):
                label = target[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
            
            accuracy = 100. * correct / total
            pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
    
    avg_loss = loss_sum / len(data_loader)
    accuracy = 100. * correct / total
    class_accuracies = {i: 100. * class_correct[i] / class_total[i] for i in range(10)}
    
    return avg_loss, accuracy, class_accuracies

# Training loop
def train():
    print("\nStarting training...")
    best_accuracy = 0
    training_history = {
        'epoch_accuracies': [],
        'epoch_losses': [],
        'best_accuracy': 0,
        'training_start': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_data = []
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update metrics
            accuracy = 100. * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
            
            if batch_idx % 10 == 9:
                epoch_data.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'step': len(epoch_data) + 1
                })
        
        # Evaluation phase
        eval_loss, eval_accuracy, class_accuracies = evaluate_model(test_loader, f'Evaluating Epoch {epoch+1}')
        
        # Update best accuracy
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Save epoch metrics
        epoch_summary = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'class_accuracies': class_accuracies
        }
        
        training_history['epoch_accuracies'].append(eval_accuracy)
        training_history['epoch_losses'].append(eval_loss)
        training_history['best_accuracy'] = best_accuracy
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.2f}%")
        print("Per-class accuracies:")
        for digit, acc in class_accuracies.items():
            print(f"Digit {digit}: {acc:.2f}%")
        
        # Save training data for visualization
        with open('static/training_data.json', 'w') as f:
            json.dump({
                'batch_data': epoch_data,
                'epoch_summary': epoch_summary,
                'training_history': training_history
            }, f)
    
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")
    torch.save(model.state_dict(), 'final_model.pth')

def test():
    print("\nStarting final evaluation...")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy, class_accuracies = evaluate_model(test_loader, 'Final Testing')
    
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print("\nPer-class accuracies:")
    for digit, acc in class_accuracies.items():
        print(f"Digit {digit}: {acc:.2f}%")
    
    # Save test results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'class_accuracies': class_accuracies
    }
    
    with open('static/test_results.json', 'w') as f:
        json.dump(results, f)

    # Generate predictions for sample images
    print("\nGenerating sample predictions...")
    random_indices = random.sample(range(len(test_dataset)), 10)
    sample_results = []
    
    for idx in random_indices:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        output = model(image)
        pred = output.argmax(dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item() * 100
        
        sample_results.append({
            'index': idx,
            'true_label': label,
            'predicted_label': pred,
            'confidence': confidence
        })
    
    with open('static/sample_results.json', 'w') as f:
        json.dump(sample_results, f)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    train()
    test()
    print("\nAll done! Check the web interface for detailed results.")