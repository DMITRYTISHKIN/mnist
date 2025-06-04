import torch
import torchvision.transforms as transforms
from PIL import Image
from mnist import ConvNet  # импорт своей модели

# 1. Подгружаем модель
model = ConvNet()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# 2. Подгружаем и обрабатываем изображение
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # если вдруг RGB
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image = Image.open("my_digit.png")  # путь к твоей картинке
image = transform(image).unsqueeze(0)  # добавляем batch размерность

# 3. Предсказание
with torch.no_grad():
    output = model(image)
    predicted = output.argmax(dim=1).item()

print(f"Предсказанная цифра: {predicted}")
