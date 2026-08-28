# Copyright 2020 Smeet Shah
#  MIT License (https://opensource.org/licenses/MIT)


import torch
import torch.nn as nn
import torch.nn.functional as F



class Visual_encoder(nn.Module):
    def __init__(self, args):
        super(Visual_encoder, self).__init__()
        self.args = args

        # visual frontend
        self.v_frontend = VisualFrontend(args)
        self.v_ds = nn.Conv1d(512, 256, 1, bias=False)

        # visual adaptor
        stacks = []
        for x in range(5):
            stacks +=[VisualConv1D(args, V=256, H=512)]
        self.visual_conv = nn.Sequential(*stacks)
            
        

    def forward(self, visual):
        visual = self.v_frontend(visual.unsqueeze(1))
        visual = self.v_ds(visual)

        visual = self.visual_conv(visual)
        return visual



class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch



class VisualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self, args):
        super(VisualFrontend, self).__init__()
        self.args =args
        if self.args.causal:
            padding = (4,3,3)
        else:
            padding = (2,3,3)

        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=padding, bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet()
        return


    def forward(self, batch):
        batchsize = batch.shape[0]

        batch = self.frontend3D[0](batch)
        if self.args.causal:
            batch = batch[:,:,:-4,:,:]
        batch = self.frontend3D[1](batch)
        batch = self.frontend3D[2](batch)
        batch = self.frontend3D[3](batch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1 ,2)
        return outputBatch




class VisualConv1D(nn.Module):
    def __init__(self, args, V=256, H=512, kernel_size=3, dilation=1):
        super(VisualConv1D, self).__init__()
        self.args =args

        self.relu_0 = nn.ReLU()
        self.norm_0 = nn.BatchNorm1d(V)
        self.conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        self.relu = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(H)
        self.dconv_pad = (dilation * (kernel_size - 1)) // 2 if not self.args.causal else (
            dilation * (kernel_size - 1))
        self.dsconv = nn.Conv1d(H, H, kernel_size, stride=1, padding=self.dconv_pad, dilation=1, groups=H)
        self.prelu = nn.PReLU()
        self.norm_2 = nn.BatchNorm1d(H)
        self.pw_conv = nn.Conv1d(H, V, 1, bias=False)

    def forward(self, x):
        out = self.relu_0(x)
        out = self.norm_0(out)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = self.norm_1(out)
        out = self.dsconv(out)
        if self.args.causal:
            out = out[:, :, :-self.dconv_pad]
        out = self.prelu(out)
        out = self.norm_2(out)
        out = self.pw_conv(out)
        return out + x


