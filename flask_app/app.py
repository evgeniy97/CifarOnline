import torch
import torch.nn as nn
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

def loadModel():
    pass

def preproccesImage():
    pass

@app.route('/imports',methods=['POST'])
def post():
    pass

if __name__ == '__main__':
    app.run()