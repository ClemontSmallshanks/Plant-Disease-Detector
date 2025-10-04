
import torch
from model import LeafNet

def load_model(pth_path="best_Leafnet.pth", num_classes=38):
    model = LeafNet(num_classes=num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")))
    model.eval()
    return model
