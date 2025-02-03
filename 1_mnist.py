import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ✅ 1. MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# ✅ 2. PyTorch 신경망(ANN) 모델 정의
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # 입력층 → 첫 번째 은닉층
        self.fc2 = nn.Linear(256, 256)  # 첫 번째 은닉층 → 두 번째 은닉층
        self.fc3 = nn.Linear(256, 10)   # 두 번째 은닉층 → 출력층
        self.relu = nn.ReLU()           # 활성화 함수

    def forward(self, x):
        x = x.view(-1, 784)  # (Batch, 28x28) → (Batch, 784) 로 변환
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ✅ 3. 모델 및 손실 함수, 옵티마이저 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Mac GPU 지원 (Metal)
model = ANN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 4. 학습 루프 (Train)
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 그래디언트 초기화
        outputs = model(images)  # 순전파 (Forward)
        loss = criterion(outputs, labels)  # 손실 함수 계산
        loss.backward()  # 역전파 (Backward)
        optimizer.step()  # 가중치 업데이트

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ✅ 5. 모델 평가 (Test)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")  # 98% 이상 기대됨