from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from reid.models.backbone import CSPDarkNetBackbone
from reid.aligned.HorizontalMaxPool2D import HorizontalMaxPool2d


class CSPDarkNet(nn.Module):
    def __init__(self, num_classes=751, loss={'softmax', 'metric'}, aligned=True):
        super(CSPDarkNet, self).__init__()
        self.loss = loss
        self.backbone = CSPDarkNetBackbone()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load('cfg/yolov8_s.pth', map_location=device)
        # 加载有限权重
        # try:
        #     for k, v in state_dict.items():
        #         if k in self.backbone.state_dict():
        #             assert self.backbone.state_dict()[
        #                        k].size() == v.size(), f"size mismatch for {k}: checkpoint size {v.size()}, model size {self.backbone.state_dict()[k].size()}"
        #             self.backbone.state_dict()[k].copy_(v)
        # except AssertionError as e:
        #     print(f"Loading stopped due to mismatch: {e}")
        try:
            # 创建一个临时字典来存储与骨干网络匹配的权重
            temp_dict = {}
            for k, v in state_dict.items():
                if k in self.backbone.state_dict() and self.backbone.state_dict()[k].size() == v.size():
                    temp_dict[k] = v
                else:
                    assert k not in self.backbone.state_dict() or self.backbone.state_dict()[
                        k].size() == v.size(), f"size mismatch for {k}: checkpoint size {v.size()}, model size {self.backbone.state_dict()[k].size()}"

            # 使用load_state_dict方法来更新骨干网络的权重
            self.backbone.load_state_dict(temp_dict, strict=False)
        except AssertionError as e:
            print(f"Loading stopped due to mismatch: {e}")
            # 可以在这里处理错误，例如记录日志、调整模型等。

        self.base = nn.Sequential(*list(self.backbone.children())[:-2])
        self.feat_dim = 2048  # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        self.classifier = nn.Linear(2048, num_classes)
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        global lf
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned:
                return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned:
                return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
