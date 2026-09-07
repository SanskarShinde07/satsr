import torch
from src.models.loader import load_pretrained

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

model = load_pretrained("checkpoints/esrgan_1S2.pth", num_ch=4)
x = torch.randn(1, 4, 130, 130)
with torch.no_grad():
    print(model(x).shape)