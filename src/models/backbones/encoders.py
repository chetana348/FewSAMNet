import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    
    def __init__(self, depth = 50, pretrained = True):
        super(ResNet, self).__init__()
        if depth == 50:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feature_dims = [256, 512, 1024, 2048]
        elif depth == 34:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.feature_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.enc0 = torch.nn.Sequential(*list(resnet.children())[:4]) 
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

    def forward(self, x):
        x = self.enc0(x)
        feat1 = self.enc1(x)
        feat2 = self.enc2(feat1)
        feat3 = self.enc3(feat2) 
        feat4 = self.enc4(feat3)
        return [feat1, feat2, feat3, feat4]


class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        features = vgg.features

        # Divide the VGG16 feature extractor into 5 stages:
        self.stage0 = nn.Sequential(*features[:5])   # conv1_1 to pool1 → out: 64 channels
        self.stage1 = nn.Sequential(*features[5:10]) # conv2_1 to pool2 → out: 128
        self.stage2 = nn.Sequential(*features[10:17])# conv3_1 to pool3 → out: 256
        self.stage3 = nn.Sequential(*features[17:24])# conv4_1 to pool4 → out: 512
        self.stage4 = nn.Sequential(*features[24:31])# conv5_1 to pool5 → out: 512

        self.feature_dims = [128, 256, 512, 512]  # skip 64 since we start from stage1 (after conv1)

    def forward(self, x):
        x = self.stage0(x)   # conv1
        f1 = self.stage1(x)  # conv2
        f2 = self.stage2(f1) # conv3
        f3 = self.stage3(f2) # conv4
        f4 = self.stage4(f3) # conv5
        return [f1, f2, f3, f4]

class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None).features

        self.enc0 = nn.Sequential(densenet.conv0, densenet.norm0, densenet.relu0, densenet.pool0)        # Output: 64
        self.enc1 = densenet.denseblock1                                                                  # Output: 256
        self.trans1 = densenet.transition1
        self.enc2 = densenet.denseblock2                                                                  # Output: 512
        self.trans2 = densenet.transition2
        self.enc3 = densenet.denseblock3                                                                  # Output: 1024
        self.trans3 = densenet.transition3
        self.enc4 = densenet.denseblock4                                                                  # Output: 1024

        self.feature_dims = [256, 512, 1024, 1024]

    def forward(self, x):
        x = self.enc0(x)
        feat1 = self.enc1(x)
        x = self.trans1(feat1)
        feat2 = self.enc2(x)
        x = self.trans2(feat2)
        feat3 = self.enc3(x)
        x = self.trans3(feat3)
        feat4 = self.enc4(x)
        return [feat1, feat2, feat3, feat4]

class EfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet, self).__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None).features

        # Extract and organize stages
        self.enc0 = nn.Sequential(*backbone[0:2])   # conv + norm + activation
        self.enc1 = nn.Sequential(*backbone[2:4])   # 16→24 channels
        self.enc2 = nn.Sequential(*backbone[4:6])   # 24→40 channels
        self.enc3 = nn.Sequential(*backbone[6:8])   # 40→112 channels
        self.enc4 = nn.Sequential(*backbone[8:])    # 112→320 channels

        self.feature_dims = [24, 40, 112, 320]

    def forward(self, x):
        x = self.enc0(x)
        feat1 = self.enc1(x)  # 1/4 resolution
        feat2 = self.enc2(feat1)  # 1/8
        feat3 = self.enc3(feat2)  # 1/16
        feat4 = self.enc4(feat3)  # 1/32
        return [feat1, feat2, feat3, feat4]


class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Encoder, self).__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None).features

        self.enc0 = nn.Sequential(base_model[0])              # Conv + BN + ReLU6
        self.enc1 = nn.Sequential(*base_model[1:4])            # 16 -> 24 (1/4 res)
        self.enc2 = nn.Sequential(*base_model[4:7])            # 24 -> 32 (1/8 res)
        self.enc3 = nn.Sequential(*base_model[7:14])           # 32 -> 96/160 (1/16 res)
        self.enc4 = nn.Sequential(*base_model[14:])            # 160 -> 320 (1/32 res)

        self.feature_dims = [24, 32, 96, 320]

    def forward(self, x):
        x = self.enc0(x)
        feat1 = self.enc1(x)     # 1/4
        feat2 = self.enc2(feat1) # 1/8
        feat3 = self.enc3(feat2) # 1/16
        feat4 = self.enc4(feat3) # 1/32
        return [feat1, feat2, feat3, feat4]


class SqueezeNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(SqueezeNetEncoder, self).__init__()
        squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None).features
        
        self.enc0 = nn.Sequential(squeezenet[0])                     # Conv1
        self.enc1 = nn.Sequential(squeezenet[1:4])                   # MaxPool + Fire2, Fire3
        self.enc2 = nn.Sequential(squeezenet[4:7])                   # Fire4, MaxPool, Fire5
        self.enc3 = nn.Sequential(squeezenet[7:10])                  # Fire6, Fire7, Fire8
        self.enc4 = nn.Sequential(squeezenet[10:])                   # MaxPool, Fire9
        
        self.feature_dims = [64, 128, 128, 256]  # Approximate dims

    def forward(self, x):
        x = self.enc0(x)       # -> [B, 64, H/2, W/2]
        feat1 = self.enc1(x)   # -> [B, 128, H/4, W/4]
        feat2 = self.enc2(feat1)  # -> [B, 128, H/8, W/8]
        feat3 = self.enc3(feat2)  # -> [B, 128, H/8, W/8]
        feat4 = self.enc4(feat3)  # -> [B, 256, H/16, W/16]
        return [feat1, feat2, feat3, feat4]
