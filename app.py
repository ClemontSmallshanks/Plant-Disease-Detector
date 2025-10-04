
from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
from load_model import load_model


app = Flask(__name__)


model = load_model("best_LeafNet.pth", num_classes=15)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])



import json
with open("class_to_idx.json") as f:
    class_to_idx = json.load(f)
disease_classes = [cls for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1])]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file:
            img = Image.open(file.stream).convert("RGB")
            img = transform(img).unsqueeze(0)  

            with torch.no_grad():
                outputs = model(img)
                _, predicted = outputs.max(1)
                class_id = predicted.item()
                prediction = disease_classes[class_id] if class_id < len(disease_classes) else str(class_id)

    return render_template("index.html", prediction=prediction)
    

if __name__ == "__main__":
    app.run(debug=True)
