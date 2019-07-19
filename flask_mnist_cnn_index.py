import flask
from flask_cors import CORS
from flask import Blueprint,render_template,send_file

import base64
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms as T
import torch
import io

app = flask.Flask(__name__)

model = None

def load_model():
    global model
    model = ConvNet()
    model.load_state_dict(torch.load('model/20_model.pth'))
    model.eval()

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out


def prepare_image(image, target_size):
    image = image.convert('L')
    # plt.imshow(image)
    # plt.show()
    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)
    # Convert to Torch.Tensor and normalize.
    image = T.Normalize((0.1307,), (0.3081,))(image)
    # Add batch_size axis.
    image = image[None]
    with torch.no_grad():
        molded_images = Variable(image)
        return molded_images
#

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    # if flask.request.method == 'POST':
    #     if flask.request.files.get("image"):
            # Read the image in PIL format
    img = flask.request.form.get("image")
    print(img)
    image = base64.b64decode(img)
    file = open('1.jpg', 'wb')
    file.write(image)
    file.close()
    image = Image.open("1.jpg")
    # print(image)
    # image = flask.request.files["image"].read()
    # image = Image.open(io.BytesIO(image))
    # image = Image.open('test_data/7_n.png')
    # Preprocess the image and prepare it for classification.
    image = prepare_image(image, target_size=(28, 28))
    preds = F.softmax(model(image), dim=1)
    #k的意思是要显示几个
    results = torch.topk(preds.cpu().data, k=5, dim=1)
    print(results)
    data['predictions'] = list()
    # 不能返回Tensor 得解析成JSON
    for prob, label in zip(results[0][0], results[1][0]):
        r = {"label": label.item(), "probability": float(prob)}
        data['predictions'].append(r)
    data["success"] = True
    return flask.jsonify(data)


@app.route('/')
def index():
   return send_file("index.html")

if __name__ == '__main__':
   load_model()
   CORS(app, supports_credentials=True)
   app.run(host='0.0.0.0')