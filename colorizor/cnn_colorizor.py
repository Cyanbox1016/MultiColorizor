import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv1d
import torch.nn.functional as F
from skimage import color
from PIL import Image
import cv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.softmax = nn.Softmax(dim=1)

        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def normalize_l(self, input):
        return (input - 50.) / 100.

    def unnormalize_ab(self, input):
    		return input * 110.
    
    def forward(self, input: torch.Tensor):
        input = input.type(torch.float32)

        conv1 = self.model1(self.normalize_l(input))
        conv2 = self.model2(conv1)
        conv3 = self.model3(conv2)
        conv4 = self.model4(conv3)
        conv5 = self.model5(conv4)
        conv6 = self.model6(conv5)
        conv7 = self.model7(conv6)
        conv8 = self.model8(conv7)
        out = self.model_out(self.softmax(conv8))

        return self.unnormalize_ab(self.upsample4(out))

def colorize(img, ckpt_pth):
    img_resized = np.asarray(Image.fromarray(img).resize((256, 256), resample=3))
    
    img_lab = color.rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_resized_lab = color.rgb2lab(img_resized)
    
    img_lum = img_lab[:, :, 0]
    img_resized_lum = img_resized_lab[:, :, 0]

    img_lum_tensor = torch.Tensor(img_lum)[None, None, :, :]
    img_resized_lum_tensor = torch.tensor(img_resized_lum)[None, None, :, :]

    model = Net().eval()
    weights = torch.load(ckpt_pth)
    model.load_state_dict(weights)
    
    out_ab = model(img_resized_lum_tensor)
    
    out_size = out_ab.shape[2:]
    img_size = img_lum_tensor.shape[2:]

    if (img_size[0] != out_size[0] or img_size[1] != out_size[1]):
        out_ab = F.interpolate(out_ab, size=img_size, mode='bilinear')
    
    out_lab = torch.cat((img_lum_tensor, out_ab), dim = 1)
    out = color.lab2rgb(out_lab.cpu().detach().numpy()[0,...].transpose((1, 2, 0)))

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imshow("title", out)
    cv2.waitKey(0)
