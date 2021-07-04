import nibabel as ni
from PIL import Image
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import torchvision
import torch
import os
from mpl_toolkits.mplot3d import Axes3D


def test():
    epi_img = ni.load('data/train/BraTS20_Training_001/BraTS20_Training_001_flair.nii')
    epi_img_data = epi_img.get_fdata()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = epi_img_data[0]



    #torchvision.utils.save_image(torch.tensor(image2d),"out/test.png",normalize=True)

def explore(i):
    f1 = f"data/train/BraTS20_Training_{i}"
    for file in os.listdir(f1):
        nii = os.path.join(f1,file)
        epi_img = ni.load(nii)
        epi_img_data = epi_img.get_fdata()
        print(epi_img_data.shape)
        print(nii)
        x = epi_img_data[:, :, 70]
        print()
        plt.imshow(x)
        plt.show()

explore("001")




