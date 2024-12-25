# MNIST CNN with Real-time Training Visualization

## Requirements
- Python 3.8+
- PyTorch
- Flask
- Matplotlib
- NumPy

## Installation
```bash
pip install torch torchvision flask matplotlib numpy
```

## Project Structure
```
mnist_cnn/
├── HowTo.md
├── train.py
├── model.py
├── server.py
├── templates/
│   └── index.html
└── static/
    └── css/
        └── style.css
```

## Steps to Run
1. Start the Flask server:
```bash
python server.py
```

2. In a new terminal, start the training:
```bash
python train.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

You'll see real-time training progress and loss curves.
After training completes, the page will display results on 10 random test images.