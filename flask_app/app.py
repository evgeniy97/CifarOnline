import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import cv2

import io
import numpy as np
from flask import Flask, request, render_template, redirect

PATH = 'model/model.torch'
app = Flask(__name__)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def loadModel(path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def preproccesImage(image):
    # Привести картинку к формату 32 на 32, затем привести к формату тензора
    img = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img)

@app.route('/', methods=["GET", "POST"])
def hello():

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print('Here:', image)
            return redirect(request.url)

    return render_template('form.html')

@app.route('/post',methods=['POST'])
def post():
    imagefile = request.files["image"]
    in_memory_file = io.BytesIO()
    imagefile.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    answer = model(preproccesImage(data))
    _, predicted = torch.max(answer.data, 1)
    print(predicted)
    return render_template("ok.html")

model = loadModel(PATH)

if __name__ == '__main__':
    app.run()