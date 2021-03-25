import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision import transforms
from torchvision import datasets
from models import CNN
from trainer import train_source_cnn
from utils import get_logger


def main(args):
    args.logdir = args.logdir+args.mode
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'train_source.log'))
    logger.info(args)

    # data
    # source_transform = transforms.Compose([
    #     transforms.ToTensor()]
    # )
    # source_dataset_train = SVHN(
    #     './input', 'train', transform=source_transform, download=True)
    # source_dataset_test = SVHN(
    #     './input', 'test', transform=source_transform, download=True)
    # source_train_loader = DataLoader(
    #     source_dataset_train, args.batch_size, shuffle=True,
    #     drop_last=True,
    #     num_workers=args.n_workers)
    # source_test_loader = DataLoader(
    #     source_dataset_test, args.batch_size, shuffle=False,
    #     num_workers=args.n_workers)
    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join('dataset', source_dataset_name)
    target_image_root = os.path.join('dataset', target_dataset_name)
    batch_size = 128
    image_size = 28
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset_source_train = datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_source,
        download=True
    )
    source_train_loader = torch.utils.data.DataLoader(
        dataset=dataset_source_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dataset_source_test = datasets.MNIST(
        root='dataset',
        train=False,
        transform=img_transform_source,
    )

    source_test_loader = torch.utils.data.DataLoader(
        dataset=dataset_source_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    # train source CNN
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        source_cnn.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    source_cnn = train_source_cnn(
        source_cnn, source_train_loader, source_test_loader,
        criterion, optimizer, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='outputs/garbage_')
    parser.add_argument('--message', '-m',  type=str, default='')
    parser.add_argument('--mode', type=str, default='s2u',choices=['m2mm','s2u'])
    args, unknown = parser.parse_known_args()
    main(args)
