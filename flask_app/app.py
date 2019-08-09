import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from flask import Flask, request, jsonify, Response

PATH = 'model.torch'
app = Flask(__name__)

def loadModel(path):
    model = torch.load(path)
    model.eval()
    return model

def preproccesImage(image):
    # Привести картинку к формату 32 на 32, затем привести к формату тензора
    img = PIL.Image.fromarray(image).resize((32,32),PIL.Image.ANTIALIAS)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img)

@app.route('/imports',methods=['POST'])
def post():
    # получать картинку
    imagefile = request.files.get('imagefile', '')
    answer = model(preproccesImage(imagefile))
    _, predicted = torch.max(answer.data, 1)
    print(predicted)
    return predicted

model = loadModel(PATH)

if __name__ == '__main__':
    app.run()