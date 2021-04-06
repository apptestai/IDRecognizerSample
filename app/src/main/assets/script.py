import torch
import cv2
import json
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


def load_segmentation_definition(path):
    mask_definition_json = json.load(open(path, 'r'))
    mask_definitions = {}
    for spec in mask_definition_json['annotation_definitions'][1]['spec']:
        label_name = spec['label_name']
        r = round(spec['pixel_value']['r'] * 255)
        g = round(spec['pixel_value']['g'] * 255)
        b = round(spec['pixel_value']['b'] * 255)
        rgb = np.array([b, g, r], dtype=np.int64)
        mask_definitions[label_name] = rgb
    return mask_definitions


dict_mask_definitions = load_segmentation_definition('D:\\Workspace\\OCR\\ID_Segmentation\\data\\Train\\annotation_definitions.json')
im = Image.open('photo.jpg')
# im = im.resize((480, 480))
T = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
input_tensor = T(im)
input_tensor = torch.unsqueeze(input_tensor, 0)
model = torch.jit.load('model.pt')
masks = model(input_tensor)['out']
masks = torch.squeeze(masks, 0)
masks = torch.argmax(masks, dim=0)
classes_mask = masks.data.cpu().numpy()
height, width = np.shape(classes_mask)

segmentation_im = np.zeros((height, width, 3), dtype=np.uint8)
for index, label_name in enumerate(dict_mask_definitions.keys()):
    rgb = dict_mask_definitions[label_name]
    indices = np.equal(classes_mask, index + 1)
    segmentation_im[indices] = rgb
cv2.imshow('segmentation', segmentation_im)
cv2.waitKey(0)
