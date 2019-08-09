import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from flask import Flask, request, render_template

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
    print(image)
    print(type(image))
    img = PIL.Image.frombytes(image).resize((32,32),PIL.Image.ANTIALIAS)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img)

@app.route('/')
def hello():
    return render_template('form.html')

@app.route('/post',methods=['POST'])
def post():
    # получать картинку
    #imagefile = request.files.get('imagefile', '')
    imagefile = request.files.get("image")
    print(type(imagefile))
    print(imagefile)
    #answer = model(preproccesImage(imagefile))
    #_, predicted = torch.max(answer.data, 1)
    #print(predicted)
    #preticted = 12
    return render_template("ok.html")

model = loadModel(PATH)

if __name__ == '__main__':
    app.run()