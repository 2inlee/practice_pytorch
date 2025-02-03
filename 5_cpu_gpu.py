import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# ✅ 1. 학습 장치 선택 (CPU & GPU 비교)
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ✅ 2. MNIST 데이터셋 로드 & 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)

# ✅ 3. CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# ✅ 4. 모델 학습 함수 (Epoch 3, 시간 측정)
def train_model(device, device_name):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3
    total_start_time = time.time()  # 전체 학습 시작 시간

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()  # 각 Epoch 시작 시간

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_time = time.time() - epoch_start_time  # Epoch 학습 시간
        accuracy = 100 * correct / total
        print(f"({device_name}) Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")

    total_time = time.time() - total_start_time  # 전체 학습 시간
    print(f"✅ ({device_name}) Total Training Time: {total_time:.2f}s\n")
    return total_time

# ✅ 5. CPU 학습 실행
print("🔹 Training on CPU...")
cpu_time = train_model(device_cpu, "CPU")

# ✅ 6. GPU 학습 실행
if device_gpu.type != "cpu":
    print("🔹 Training on GPU...")
    gpu_time = train_model(device_gpu, "GPU")
else:
    print("⚠ No GPU detected. Skipping GPU training.")
    gpu_time = None

# ✅ 7. 결과 비교
if gpu_time:
    speedup = cpu_time / gpu_time
    print(f"🚀 Speedup (CPU → GPU): {speedup:.2f}x faster on GPU!")