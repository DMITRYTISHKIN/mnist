import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

def main():
    # ✅ Гиперпараметры
    batch_size = 64
    learning_rate = 0.1
    epochs = 20

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

    # ✅ Инициализация модели, loss и optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,   # каждые 10 эпох
        gamma=0.1       # уменьшать lr в 10 раз
    )

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    lrs = []

    # ✅ Обучение
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()             # обнуляем градиенты
            outputs = model(images)           # прямой проход
            loss = criterion(outputs, labels) # считаем ошибку
            loss.backward()                   # обратное распространение
            optimizer.step()                  # обновление весов
            scheduler.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)   # argmax по классам
            total += labels.size(0)                    # сколько всего образцов
            correct += (predicted == labels).sum().item()  # сколько угадано

        lrs.append(optimizer.param_groups[0]['lr'])
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

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

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Точность на тесте: {correct / total:.2f}%")

    torch.save(model.state_dict(), "mnist_cnn.pth")

    # ✅ Вывод графика
    fig, ax1 = plt.subplots()

    # Левая ось Y — Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(train_losses, color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Правая ось Y — Accuracy
    ax2 = ax1.twinx()  # вторая ось Y
    color = 'tab:blue'
    ax2.set_ylabel('Train Accuracy', color=color)
    ax2.plot(train_accuracies, color=color, marker='x', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Заголовок и сетка
    plt.title("Train Loss & Accuracy per Epoch")
    fig.tight_layout()
    plt.grid(True)

    # Сохраняем
    plt.savefig("train_loss_and_accuracy.png")
    plt.show()
    plt.clf()


    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy per Epoch")
    plt.grid()
    plt.legend()
    plt.savefig("accuracy_comparison.png")
    plt.show()
    plt.clf()

    plt.plot(lrs)
    plt.title("Learning Rate per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.savefig("learning_rate_schedule.png")
    plt.show()

if __name__ == "__main__":
    main()