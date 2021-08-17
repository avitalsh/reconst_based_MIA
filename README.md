# Membership Inference Attacks are Easier on Difficult Problems

> [Membership Inference Attacks are Easier on Difficult Problems](https://arxiv.org/abs/2102.07762)  
> Avital Shafran, Shmuel Peleg and Yedid Hoshen  
> International Conference on Computer Vision (ICCV), 2021.



## Usage
The current software is tested with Pytorch 1.6.0 and Python 3.6.

### Dataset
Download the Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/) (registration required).
After downloading, please put it under the `./pix2pixHD/datasets` folder in the same way the example images are provided.

### Pre-trained model
Please download the pre-trained pix2pixHD Cityscapes model from [the official pix2pixHD Pytorch implementation](https://github.com/NVIDIA/pix2pixHD), and put it under `./pix2pixHD/checkpoints/label2city_1024p/`

### Run attack
Run attack on pre-trained pix2pixHD model:
```
python run_MIA_pix2pixHD.py --dataroot./pix2pixHD/datasets/Cityscapes --checkpoints_dir ./pix2pixHD/checkpoints
```

## Citation

If you find this project useful for your research, please cite

```
@article{shafran2021reconstruction,
  title={Reconstruction-Based Membership Inference Attacks are Easier on Difficult Problems},
  author={Shafran, Avital and Peleg, Shmuel and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2102.07762},
  year={2021}
}
```

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
