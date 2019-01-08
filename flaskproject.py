import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from scipy import misc
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import flask
from flask import Flask,render_template,request

#defining cat-to-name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#defines the structures
structures = {"vgg16":25088,
              "densenet121" : 1024,
              "alexnet" : 9216 }
#defines the setup
def nn_setup(structure='vgg16',dropout=0.5, hidden_layer1 = 120,lr = 0.001):
    
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))
        
    
        
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        model.cpu()
        
        return model , optimizer ,criterion 

#loads the model
def load_model(path):
    checkpoint = torch.load(path,map_location='cpu')
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_ = nn_setup(structure , 0.5,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict']) 
    return model
model = load_model('best_model.pth')

def process_image(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=1):   
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file = request.files['image']
        if not file: 
            return render_template('index.html', label="No file")
        prediction = predict(file,model)
        label = str(np.squeeze(prediction))
        if label=='10': 
            label='0'
        return render_template('index.html', label=label, file=file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    model = load_model('best_model.pth')