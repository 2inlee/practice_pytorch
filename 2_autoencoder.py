import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ✅ 1. MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # 28x28 -> 784로 Flatten

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# ✅ 2. PyTorch AutoEncoder 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 인코딩(Encoding): 784 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        # 디코딩(Decoding): 128 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ✅ 3. 모델 및 손실 함수, 옵티마이저 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Mac GPU 지원 (Metal)
model = AutoEncoder().to(device)

criterion = nn.MSELoss()  # Mean Squared Error (MSE)
optimizer = optim.RMSprop(model.parameters(), lr=0.02)

# ✅ 4. 학습 루프 (Train)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, _ in train_loader:  # AutoEncoder는 레이블(y)이 필요 없음
        images = images.to(device)

        optimizer.zero_grad()  # 그래디언트 초기화
        outputs = model(images)  # 순전파 (Forward)
        loss = criterion(outputs, images)  # 손실 함수 계산
        loss.backward()  # 역전파 (Backward)
        optimizer.step()  # 가중치 업데이트

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ✅ 5. 모델 평가 (Test)
model.eval()
examples_to_show = 10
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)
    reconstructed = model(images)

# ✅ 6. 원본 MNIST 데이터 vs. Reconstruction 결과 비교
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(images[i].cpu().view(28, 28).numpy(), cmap='gray')
    a[1][i].imshow(reconstructed[i].cpu().view(28, 28).numpy(), cmap='gray')

f.savefig('reconstructed_mnist_image.png')  # reconstruction 결과를 png로 저장
f.show()
plt.draw()
plt.waitforbuttonpress()