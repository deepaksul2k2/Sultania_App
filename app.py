import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize model
model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save model (.pkl)
def save_model(path="mnist_cnn.pkl"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model (.pkl)
def load_model(path="mnist_cnn.pkl"):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {path}")

if __name__ == "__main__":
    train()
    test()
    save_model()

# -------------------- FASTAPI INFERENCE API --------------------
# Save this part as api.py or keep in same file and run separately

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model once at startup
api_model = CNN().to(DEVICE)
api_model.load_state_dict(torch.load("mnist_cnn.pkl", map_location=DEVICE))
api_model.eval()

class InputData(BaseModel):
    pixels: list  # Expecting flattened 28x28 = 784 values

@app.get("/")
def home():
    return {"message": "MNIST CNN API is running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to tensor
        img = np.array(data.pixels).reshape(28, 28).astype(np.float32)
        img = (img / 255.0 - 0.1307) / 0.3081

        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = api_model(tensor)
            _, predicted = torch.max(outputs, 1)

        return {"prediction": int(predicted.item())}

    except Exception as e:
        return {"error": str(e)}

# To run:
# uvicorn api:app --reload

