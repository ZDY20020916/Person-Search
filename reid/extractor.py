from sklearn.preprocessing import normalize
from reid import models
import torch
from torchvision import transforms
from reid.util.utils import *


def getFeature(img):
    reidmodel = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=False,
                                  aligned=True)
    checkpoint = torch.load("reid/log/checkpoint_ep300.pth.tar")
    reidmodel.load_state_dict(checkpoint['state_dict'])

    img_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # reidmodel = reidmodel.cuda()
    reidmodel.eval()

    img = img_to_tensor(img, img_transform)
    f, lf = reidmodel(img)
    print('ok')
    a = f / (torch.norm(f, p=2, keepdim=True) + 1e-12)
    # return a.cpu().detach().numpy()
    return a.detach().numpy()


def rank(f, f_arrays):
    distances = []
    # 计算每个F与f的欧氏距离，并将id和距离存储为元组
    for i, F in f_arrays:
        distance = np.linalg.norm(f - F)
        distances.append((i, distance))
    # 对距离进行排序（按照距离升序），同时保持id的对应关系
    sorted_distances = sorted(distances, key=lambda x: x[1])[:5]
    # 输出排序后距离最小的序号
    ret = []
    for i, distance in sorted_distances:
        ret.append(i)
    return ret
