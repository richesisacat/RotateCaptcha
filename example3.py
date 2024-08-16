import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 360)  # 360 classes for 0-358 degrees rotation

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
model = CNN()
model.load_state_dict(torch.load('rotate_model_3.pth'))
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
])


# Function to predict rotation angle
def predict_rotation_angle(image_path, model, transform):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_angle = predicted.item()  # Get the predicted angle
    return predicted_angle


def rotate_image(image_path, degrees_to_rotate, output_path):
    # 打开图像
    image = Image.open(image_path)

    print(degrees_to_rotate)
    # 旋转图像
    rotated_image = image.rotate(-degrees_to_rotate)

    # 保存旋转后的图像
    rotated_image.save(output_path)

    print("图像已旋转并保存到:", output_path)


if __name__ == '__main__':
    path = 'D:\\dba\\image\\'
    # name = 'w230.0_image_1723424789.png'
    # image_path = path + name
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            predicted_angle = predict_rotation_angle(full_path, model, transform)
            rotate_image(full_path, predicted_angle, path + 'new\\' + name)
