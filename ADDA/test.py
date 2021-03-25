from logging import getLogger
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import torch
from utils import AverageMeter, save

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms

from models import CNN, Discriminator
from trainer import train_target_cnn
from utils import get_logger





def step(model, data, target, criterion, args):
    data, target = data.to(args.device), target.to(args.device)
    output = model(data)
    loss = criterion(output, target)
    return output, loss



def validate(model, dataloader, criterion, args=None):
    model.eval()
    losses = AverageMeter()
    targets, probas = [], []
    with torch.no_grad():
        for iter_i, (data, target) in enumerate(dataloader):
            bs = target.size(0)
            output, loss = step(model, data, target, criterion, args)
            output = torch.softmax(output, dim=1)  # NOTE: check
            losses.update(loss.item(), bs)
            targets.extend(target.cpu().numpy().tolist())
            probas.extend(output.cpu().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)
    acc = accuracy_score(targets, preds)
    return {
        'loss': losses.avg, 'acc': acc,
    }

class ARG:
    def __init__(self):
        self.device = 'cuda:0'

if __name__ == "__main__":
    args = ARG()
    target_cnn = CNN(in_channels=3, target=True).to(args.device)
    c = torch.load('outputs/garbage1/best_model.pt')
    target_cnn.load_state_dict(c['model'])
    target_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    criterion = nn.CrossEntropyLoss()
    target_dataset_test = MNIST(
        './input', train=False, transform=target_transform, download=True)
    target_test_loader = DataLoader(
        target_dataset_test, 128, shuffle=False,
        num_workers=4)
    print(len(target_test_loader))
    validation = validate(
        target_cnn, target_test_loader, criterion, args=args)
    print(validation)