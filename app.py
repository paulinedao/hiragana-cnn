# app.py
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64


server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Define model

class HiraganaCNN(nn.Module):
    def __init__(self, num_classes=46):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# Load model
checkpoint = torch.load("hiragana_cnn_model.pth", map_location="cpu")
class_names = checkpoint["class_names"]
model = HiraganaCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.eval()


 
# Transform
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(img: Image.Image):
    img = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
    return class_names[pred_idx], probs[0][pred_idx].item()

# app layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Hiragana Character Recognition", className="text-center my-4"),
    
    dcc.Upload(
        id="upload-image",
        children=html.Div(["Drag and Drop or ", html.A("Select an Image")]),
        accept="image/*",
        multiple=False,
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin-bottom": "20px"
        }
    ),
    
    html.Div(id="image-container", className="text-center"),
    html.H4(id="prediction-output", className="text-success mt-3 text-center")
], fluid=True)


@app.callback(
    Output("image-container", "children"),
    Output("prediction-output", "children"),
    Input("upload-image", "contents")
)
def handle_upload(contents):
    if contents is None:
        return "", ""
    
    # Convert uploaded file to PIL image
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert("L")
    
    # Predict
    pred, conf = predict_image(img)
    
    # Display image
    img_element = html.Img(src=contents, style={"width": "200px", "margin-top": "10px"})
    result_text = f"Prediction: {pred} ({conf:.2%})"
    
    return img_element, result_text


if __name__ == "__main__":
    app.run(debug=True)
