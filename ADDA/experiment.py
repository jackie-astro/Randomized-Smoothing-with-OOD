import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from torchvision import datasets
from models import CNN, Discriminator
from trainer import train_target_cnn
from utils import get_logger
from data_loader import GetLoader
from data_load import svhn, usps

def run(args):
    args.logdir = args.logdir+args.mode
    args.trained = args.trained+args.mode+'/best_model.pt'
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info(args)

    # data
    # source_transform = transforms.Compose([
    #     # transforms.Grayscale(),
    #     transforms.ToTensor()]
    # )
    # target_transform = transforms.Compose([
    #     transforms.Resize(32),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    # ])
    # source_dataset_train = SVHN(
    #     './input', 'train', transform=source_transform, download=True)
    # target_dataset_train = MNIST(
    #     './input', train=True, transform=target_transform, download=True)
    # target_dataset_test = MNIST(
    #     './input', train=False, transform=target_transform, download=True)
    # source_train_loader = DataLoader(
    #     source_dataset_train, args.batch_size, shuffle=True,
    #     drop_last=True,
    #     num_workers=args.n_workers)
    # target_train_loader = DataLoader(
    #     target_dataset_train, args.batch_size, shuffle=True,
    #     drop_last=True,
    #     num_workers=args.n_workers)
    # target_test_loader = DataLoader(
    #     target_dataset_test, args.batch_size, shuffle=False,
    #     num_workers=args.n_workers)
    batch_size = 128
    if args.mode == 'm2mm':
        source_dataset_name = 'MNIST'
        target_dataset_name = 'mnist_m'
        source_image_root = os.path.join('dataset', source_dataset_name)
        target_image_root = os.path.join('dataset', target_dataset_name)
        image_size = 28
        img_transform_source = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        img_transform_target = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        dataset_source = datasets.MNIST(
            root='dataset',
            train=True,
            transform=img_transform_source,
            download=True
        )


        train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

        dataset_target_train = GetLoader(
            data_root=os.path.join(target_image_root, 'mnist_m_train'),
            data_list=train_list,
            transform=img_transform_target
        )

        test_list = os.path.join(target_image_root, 'mnist_m_test_labels.txt')

        dataset_target_test = GetLoader(
            data_root=os.path.join(target_image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    elif args.mode == 's2u':
        dataset_source  = svhn.SVHN('./data/svhn/', split='train', download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(28),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))

        dataset_target_train = usps.USPS('./data/usps/', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ]))
        dataset_target_test = usps.USPS('./data/usps/', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
        source_dataset_name = 'svhn'
        target_dataset_name = 'usps'

    source_train_loader = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    target_train_loader = torch.utils.data.DataLoader(
        dataset=dataset_target_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    target_test_loader = torch.utils.data.DataLoader(
        dataset=dataset_target_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    # train source CNN
    source_cnn = CNN(in_channels=args.in_channels).to(args.device)
    if os.path.isfile(args.trained):
        print("load model")
        c = torch.load(args.trained)
        source_cnn.load_state_dict(c['model'])
        logger.info('Loaded `{}`'.format(args.trained))
    else:
        print("not load model")

    # train target CNN
    target_cnn = CNN(in_channels=args.in_channels, target=True).to(args.device)
    target_cnn.load_state_dict(source_cnn.state_dict())
    discriminator = Discriminator(args=args).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        target_cnn.encoder.parameters(),
        lr=args.lr)
    # optimizer = optim.Adam(
    #     target_cnn.encoder.parameters(),
    #     lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.lr)
    # d_optimizer = optim.Adam(
    #     discriminator.parameters(),
    #     lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    train_target_cnn(
        source_cnn, target_cnn, discriminator,
        criterion, optimizer, d_optimizer,
        source_train_loader, target_train_loader, target_test_loader,
        args=args)
