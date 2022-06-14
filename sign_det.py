import torch
import pandas
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)
def find_sign(image):
    results = model(image)
    amount = len(results.pandas().xyxy[0].index)
    image = np.squeeze(results.render())
    return image, amount