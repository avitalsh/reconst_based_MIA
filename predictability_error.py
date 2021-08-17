import os, sys

from sklearn.linear_model import LinearRegression


import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data

import copy
import shutil

from skimage.transform import resize


from sklearn.model_selection import train_test_split

from PIL import Image
import resnet


resnet_transform_color = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

resnet_transform_gray = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def extract_features(inp, wide_resnet):

    with torch.no_grad():
        _, x77, x1414, x2828, x5656 = wide_resnet(inp.cuda())

    # x = x.data.cpu().numpy()
    x77 = x77.data.cpu().numpy()
    x1414 = x1414.data.cpu().numpy()
    x2828 = x2828.data.cpu().numpy()
    x5656 = x5656.data.cpu().numpy()

    x5656 = resize(x5656, (1, 256, 56, 56), order=1)
    x2828 = resize(x2828, (1, 512, 56, 56), order=1)
    x1414 = resize(x1414, (1, 1024, 56, 56), order=1)
    x77 = resize(x77, (1, 2048, 56, 56), order=1)


    feat = np.concatenate((x5656, x2828, x1414, x77), axis=1).squeeze()
    return torch.from_numpy(feat)


def split_tr_ts(features, pixels, train_ratio=0.7):
    # if len(features.shape) == 4:
    #     assert features.shape[0] == 1
    #     features = features.squeeze()

    features = features.reshape(features.shape[0], -1)
    pixels = pixels.reshape(pixels.shape[0], -1)


    tr_idx, ts_idx = train_test_split(np.arange(features.shape[1]), train_size=train_ratio, test_size=1-train_ratio)


    tr_feat = torch.cat([features[:, i].unsqueeze(0) for i in tr_idx], axis=0)
    tr_pix = torch.cat([pixels[:, i].unsqueeze(0) for i in tr_idx], axis=0)
    ts_feat = torch.cat([features[:, i].unsqueeze(0) for i in ts_idx], axis=0)
    ts_pix = torch.cat([pixels[:, i].unsqueeze(0) for i in ts_idx], axis=0)

    if train_ratio == 1.0:
        ts_feat = tr_feat
        ts_pix = tr_pix


    features = [tr_feat, ts_feat]
    pixels = [tr_pix, ts_pix]


    return features, pixels #, tr_idx, val_idx




def compute_pred_error(inp_img_path, gt_img_path, ratio=0.7, wide_resnet=None, inp_mode='inp'):
    if wide_resnet == None:
        wide_resnet = resnet.wide_resnet50_2(pretrained=True)
    wide_resnet.to('cuda')
    wide_resnet.eval()

    gt_img = Image.open(gt_img_path)
    inp = Image.open(inp_img_path)

    if inp_mode == 'out':
        inp = gt_img

    if inp.mode == 'RGB':
        inp = resnet_transform_color(inp).unsqueeze(0)
    else:
        inp = resnet_transform_gray(inp).unsqueeze(0)

    out = transforms.ToTensor()(gt_img.resize(size=(56, 56)))

    inp_resnet_featuers = extract_features(inp, wide_resnet)


    features, pixels = split_tr_ts(inp_resnet_featuers, out, train_ratio=ratio)

    tr_f, tr_p = features[0].data.cpu().numpy(), pixels[0].data.cpu().numpy()
    ts_f, ts_p = features[1].data.cpu().numpy(), pixels[1].data.cpu().numpy()

    pix_predictor = LinearRegression()
    pix_predictor.fit(tr_f, tr_p)

    pred_tr = pix_predictor.predict(tr_f)
    pred_p = pix_predictor.predict(ts_f)

    # if dist == 'l1':
    l1_dist = np.linalg.norm(pred_p.flatten()-ts_p.flatten(), ord=1) / pred_p.size

    return l1_dist
