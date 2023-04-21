from torch.nn import TransformerEncoderLayer

import torch,os
import torch.nn as nn
import numpy as np

from einops import rearrange
import cv2

def print_rate(model_path):
    for k,v in torch.load(model_path).items():
        if "rate" in k:
            print(k,v)

label_path = "/Users/peco/Desktop/bcsformer/fig_res/mass/test_patches/masks/"
build_label_path = "/Users/peco/datasets/potsdam/build_label/"
build_mask_path = "/Users/peco/Desktop/bcsformer/fig_res/mass/test_patches/label/"

class LRDU(nn.Module):
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            #nn.Conv2d(in_c,self.factor*in_c,1,1,0), # 1.0737 G 0.0666 M
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x

if __name__ == "__main__":

    #x = torch.ones([1,64,128,128])
    #model = LRDU(64,4)

    #print(model(x).shape)

    if 0:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, x)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 


    if 1:
        for img in os.listdir(label_path):
            if img[0] !=".":
                mask = cv2.imread(label_path+img,cv2.IMREAD_UNCHANGED)

                print(mask)

                h, w = mask.shape[0], mask.shape[1]
                label = np.zeros(shape=(h, w), dtype=np.uint8)
                label[mask == 0] = 255

                cv2.imwrite(build_mask_path+img,label)


   