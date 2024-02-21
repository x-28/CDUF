import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
#from metric import sake_metric
#
def get_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        self.split = split

        images,self.refs,self.skrefs = [], {},{}
        for classes in os.listdir(os.path.join(data_root, data_name, split, 'sketch')):
            sketches = glob.glob(os.path.join(data_root, data_name, split, 'sketch', str(classes), '*.[jp][pn]g'))
            photos = glob.glob(os.path.join(data_root, data_name, split, 'photo', str(classes), '*.[jp][pn]g'))
            #val_sketches = [glob.glob(os.path.join(data_root, data_name, split, 'sketch', str(classes), '*.[jp][pn]g'))[1]]
            self.skrefs[str(classes)] = sketches
            images += sketches
            if split == 'val':
                images += photos
            else:
                self.refs[str(classes)] = photos

        self.images = sorted(images)
        self.transform = get_transform()

        self.domains, self.labels, self.classes = [], [], {}
        i = 0
        for img in self.images:
            domain, label = os.path.dirname(img).split('\\')[-2:]
            self.domains.append(0 if domain == 'photo' else 1)
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])

        self.names = {}
        for key, value in self.classes.items():
            self.names[value] = key

    def __getitem__(self, index):
        img_name = self.images[index]
        domain = self.domains[index]
        label = self.labels[index]
        img = self.transform(Image.open(img_name))
        total_class = sorted(set(self.classes.keys()))
        available_classes = total_class[:label] + total_class[label + 1:]
        neg_label = random.randrange(len(available_classes))
        neg_name = self.names[neg_label]
        neg_dir = np.random.choice(self.skrefs[self.names[neg_label]])
        neg_img = self.transform(Image.open(neg_dir))
        if self.split == 'train':
            dirname1 = os.path.basename(os.path.normpath(self.names[label]))
            pos_name = np.random.choice(self.refs[dirname1])
            dirname2 = os.path.basename(os.path.normpath(self.names[neg_label]))
            neg_name = np.random.choice(self.refs[dirname2])
            pos = self.transform(Image.open(pos_name))
            neg = self.transform(Image.open(neg_name))
            return img, neg_img, pos, neg, label,neg_label,neg_label
        else:
            return img, domain, label

    def __len__(self):
        return len(self.images)


def compute_metric(vectors, domains, labels):
    acc = {}
    sketch_vectors, photo_vectors = vectors[domains == 1], vectors[domains == 0]
    sketch_labels, photo_labels = labels[domains == 1], labels[domains == 0]

    precs_100, precs_200, maps_200, maps_all,accs_1,accs_5,accs_10 = 0, 0, 0, 0,0,0,0
    for sketch_vector, sketch_label in zip(sketch_vectors, sketch_labels):
        sim = F.cosine_similarity(sketch_vector.unsqueeze(dim=0), photo_vectors).squeeze(dim=0)
        target = torch.zeros_like(sim, dtype=torch.bool)
        target[sketch_label == photo_labels] = True
        count_true = torch.sum(target).item()
        precs_100 += retrieval_precision(sim, target, top_k=100).item()
        precs_200 += retrieval_precision(sim, target, top_k=200).item()
        maps_200 += retrieval_average_precision(sim, target,top_k=200).item()
        maps_all += retrieval_average_precision(sim, target,top_k=sim.shape[-1]).item()
        accs_1 += retrieval_accuracy(sim, target, top_k=1).item()
        accs_5 += retrieval_accuracy(sim, target, top_k=5).item()
        accs_10 += retrieval_accuracy(sim, target, top_k=10).item()
    prec_100 = precs_100 / sketch_vectors.shape[0]
    prec_200 = precs_200 / sketch_vectors.shape[0]
    map_200 = maps_200 / sketch_vectors.shape[0]
    map_all = maps_all / sketch_vectors.shape[0]
    acc_1 = accs_1 / sketch_vectors.shape[0]
    acc_5 = accs_5 / sketch_vectors.shape[0]
    acc_10 = accs_10 / sketch_vectors.shape[0]
    acc['P@100'], acc['P@200'], acc['mAP@200'], acc['mAP@all'],acc['acc@1'],acc['acc@5'] ,acc['acc@10']\
        = prec_100, prec_200, map_200, map_all,acc_1,acc_5,acc_10
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['acc@1'] + acc['acc@5'] + acc['acc@10'] ) / 3
    return acc

def retrieval_precision(sim, target, top_k=100):
    relevant = target[sim.topk(min(top_k, sim.shape[-1]), dim=-1)[1]].sum().float()
    return relevant / top_k

def retrieval_accuracy(sim, target, top_k=100):
    relevant = target[sim.topk(min(top_k, sim.shape[-1]), dim=-1)[1]].sum().float()
    return relevant

# def retrieval_accuracy(sim, target, top_k=100):
#     relevant = target[sim.topk(min(top_k, sim.shape[-1]), dim=-1)[1]].sum().float()
#     if relevant > 0:
#         acc = 1.0
#     else:
#         acc = 0.0
#     return acc

def retrieval_average_precision(sim, target, top_k=100):
    target = target[sim.topk(min(top_k, sim.shape[-1]), sorted=True, dim=-1)[1]]
    positions = torch.arange(1, len(target) + 1, device=target.device, dtype=torch.float32)[target > 0]
    d=torch.arange(1, len(target) + 1, device=target.device, dtype=torch.float32)
    e=d[target > 0]
    f = torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1
    # 处理 e 为空的情况,可以选择返回一个特定的值或者抛出一个异常
    if len(e) > 0:
        g = torch.div(f, e)
        g_mean = g.mean()
    else:
        g= f-f
        g_mean = torch.zeros(1)
    return g_mean

def parse_args():
    parser = argparse.ArgumentParser(description='Train/Test Model')
    # common args
    parser.add_argument('--data_root', default='D:/lixue/sketch-image/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='ChairV2', type=str, choices=['sketchy', 'ShoeV2','ChairV2'],
                        help='Dataset name')
    parser.add_argument('--prompt_num', default=3, type=int, help='Number of prompt embedding')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str, help='Mode of the script')

    # train args
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=60, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--triplet_margin', default=0.3, type=float, help='Margin of triplet loss')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='Learning rate of encoder')
    parser.add_argument('--prompt_lr', default=1e-3, type=float, help='Learning rate of prompt embedding')
    parser.add_argument('--cls_weight', default=0.5, type=float, help='Weight of classification loss')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (-1 for no manual seed)')

    # test args
    parser.add_argument('--query_name', default='/home/data/sketchy/val/sketch/cow/n01887787_591-14.jpg', type=str,
                        help='Query image path')
    parser.add_argument('--retrieval_num', default=8, type=int, help='Number of retrieved images')

    args = parser.parse_args()
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    return args