import os, sys
sys.path.append('./pix2pixHD')

from pix2pixHD.options.test_options import TestOptions
from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.models.models import create_model
import pix2pixHD.util.util as util

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from PIL import Image

import resnet
from predictability_error import compute_pred_error


def run_dataset(opt, model, wide_resnet, dataset, num_gen, res_dir):
    gt_reconst, pred_error = [], []

    for i, data in enumerate(dataset):
        print("image {}/{}".format(i+1, num_gen))#, end='\r')
        if i == 0:
            print(data['path'])
        if i >= num_gen:
            break


        generated = model.inference(data['label'],
                                    data['inst'],
                                    data['image'])

        real_B = np.array(Image.open(data['gt_path'][0])) /255.
        fake_B = util.tensor2im(generated.data[0]) / 255.


        gt_reconst.append(np.linalg.norm(real_B.flatten()-fake_B.flatten(), ord=1) / real_B.size)
        pred_error.append(compute_pred_error(data['path'][0], data['gt_path'][0], ratio=0.7, wide_resnet=wide_resnet))


        if i < 5:
            #for debug, plot some results to make sure they look reasonable
            img_data = {'A': util.tensor2label(data['label'][0], opt.label_nc),
                        'real_B': real_B,
                        'fake_B': fake_B}
            name = data['path'][0].split('/')[-1].split('_gtFine')[0]
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for idx, key in enumerate(img_data):
                axs[idx].set_title(key)
                axs[idx].imshow(img_data[key])
                axs[idx].set_xticks([])
                axs[idx].set_yticks([])
            plt.savefig(os.path.join(res_dir, '{}.png'.format(name)))
            plt.close()



    return np.array(gt_reconst), np.array(pred_error)

if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = False  # shuffle
    opt.no_flip = True  # no flip


    opt.phase = 'train'
    train_data_loader = CreateDataLoader(opt)
    train_dataset = train_data_loader.load_data()

    opt.phase = 'test'
    test_data_loader = CreateDataLoader(opt)
    test_dataset = test_data_loader.load_data()


    #sample equal number of train and test images
    num_gen = min(len(train_dataset), len(test_dataset))

    model = create_model(opt)

    wide_resnet = resnet.wide_resnet50_2(pretrained=True)
    wide_resnet.to('cuda')
    wide_resnet.eval()


    print("Run MIA on train images")
    train_res_dir = "./results/{}/train".format(opt.name)
    os.makedirs(train_res_dir, exist_ok=True)
    tr_gt_reconst, tr_pred_error = run_dataset(opt, model, wide_resnet, train_dataset, num_gen, train_res_dir)

    print("Run MIA on test images")
    test_res_dir = "./results/{}/test".format(opt.name)
    os.makedirs(test_res_dir, exist_ok=True)
    ts_gt_reconst, ts_pred_error = run_dataset(opt, model, wide_resnet, test_dataset, num_gen, test_res_dir)


    print("measure ROC")
    labels = [0] * len(tr_gt_reconst) + [1] * len(ts_gt_reconst)
    r = np.concatenate([tr_gt_reconst, ts_gt_reconst], axis=0)
    roc = roc_auc_score(labels, r)
    print("gt reconst ROC: {}".format(roc))
    r = np.concatenate([tr_gt_reconst - tr_pred_error, ts_gt_reconst - ts_pred_error], axis=0)
    roc = roc_auc_score(labels, r)
    print("gt reconst - pred_error ROC: {}".format(roc))

