import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ Гиперпараметры
batch_size = 64
learning_rate = 0.01
epochs = 5

# ✅ Преобразования для изображений
transform = transforms.Compose([
    transforms.ToTensor(),                             # Преобразуем в тензор
    transforms.Normalize((0.1307,), (0.3081,))          # Нормализация
])

# ✅ Загрузка MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ✅ Определим сверточную нейросеть
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Свертка 1: 1 входной канал (ч/б), 8 выходных фильтров, ядро 3x3
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # Свертка 2: 8 каналов → 16, ядро 3x3
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Пулинг
        self.pool = nn.MaxPool2d(2, 2)  # уменьшает в 2 раза
        # Dropout
        self.dropout = nn.Dropout(0.25)
        # Полносвязные слои
        self.fc1 = nn.Linear(16 * 7 * 7, 128)  # после двух пуллингов: 28→14→7
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [1×28×28] → [8×14×14]
        x = self.pool(F.relu(self.conv2(x)))   # [8×14×14] → [16×7×7]
        x = self.dropout(x)
        x = x.view(-1, 16 * 7 * 7)             # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # логиты
        return x

# ✅ Инициализация модели, loss и optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
train_losses = []

# ✅ Обучение
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # обнуляем градиенты
        outputs = model(images)           # прямой проход
        loss = criterion(outputs, labels) # считаем ошибку
        loss.backward()                   # обратное распространение
        optimizer.step()                  # обновление весов

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f}")

# ✅ Вывод графика
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Loss curve")
plt.grid()
plt.show()

# ✅ Тестирование
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

print(f"Точность на тесте: {100 * correct / total:.2f}%")
