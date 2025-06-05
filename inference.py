import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np
from mnist import ConvNet  # замените на ваш класс, если называется иначе

# Загружаем обученную модель
model = ConvNet()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

# Преобразования, как при обучении MNIST
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Предсказание
def predict(image: np.ndarray):
    if image is None:
        return "Нет изображения"
    image = Image.fromarray(image.astype("uint8")).convert("L")  # grayscale
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(1).item()
    return f"Предсказано: {prediction}"

# Интерфейс с рисованием
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        source="canvas",            # включаем рисование
        shape=(280, 280),           # удобно рисовать мышкой
        image_mode="L",             # grayscale
        invert_colors=True,         # белое на чёрном (как MNIST)
        type="numpy"
    ),
    outputs="text",
    title="MNIST Распознавалка",
    description="Нарисуй цифру от 0 до 9, и модель предскажет, что это"
)

demo.launch()
