# evaluate a smoothed classifier on a dataset
import argparse
import os
# from datasets import get_dataset, DATASETS, get_num_classes
from core import Noise
from time import time
import torch
import datetime
from model import CNNModel
import torch.backends.cudnn as cudnn
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
# from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default="cifar10", help="which dataset")
parser.add_argument("--base_classifier", default="log/checkpoint.pth.tar", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", default="0.0", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", default="certification.txt", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=1)
parser.add_argument("--N", type=int, default=500, help="number of samples to use")
parser.add_argument("--alpha1", type=float, default=0.001, help="failure probability")
parser.add_argument("--alpha2", type=float, default=0, help="failure probability")
args = parser.parse_args()


#python code/certify.py imagenet model_output_dir/checkpoint.pth.tar 0.50 certification_output
# --alpha 0.001 --N0 100 --N 100000 --skip 100 --batch 400

# r            0      0.5     1    1.5    2     2.5    3.0
#sigma  0.25   0.377  0.294   0     0     0      0      0
#       0.5    0.266  0.185   0.114  0    0      0      0
#        1     0.138  0.116   0.092 0.048 0.014  0      0



# r            0      0.5     1    1.5    2     2.5    3.0
#sigma  0.25   0.377  0.294   0     0     0      0      0
#       0.5    0.266  0.185   0.114  0    0      0      0
#        1     0.863  0.809   0.704 0.607 0.452  0      0

#mnist_m
# r            0      0.5     1    1.5    2     2.5    3.0
#sigma  0.25   0.852  0.365   0     0     0      0      0
#       0.5    0.403  0.125   0.021  0    0      0      0
#        1     0.033  0.009   0.005 0.003 0.0  0      0


if __name__ == "__main__":
    # load the base classifier
    dataset_name = "mnist_m"
    assert dataset_name in ['MNIST', 'mnist_m']

    if dataset_name == 'mnist_m':
        model_root = 'models_m'
    else:
        model_root = 'models'
    image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = False
    batch_size = 1
    image_size = 28
    alpha = 0

    """load data"""

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

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        dataset = datasets.MNIST(
            root='dataset',
            train=False,
            transform=img_transform_source,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_best.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)


    # create the smooothed classifier g
    noised_classifier = Noise(my_net, 10, args.sigma)
    print(args.sigma)
    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset

    n_total = 0
    n_correct = 0
    thresh_list = [0,0.5,1.0,1.5,2.0,2.5,3.0]
    correct_list = [0]*7
    while n_total < 1000:
        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
        prediction, radius = noised_classifier.certify(t_img, args.N0, args.N, args.alpha1, args.alpha2, batch_size)
        t_label = t_label.numpy()
        print(n_total, t_label[0], prediction, radius)
        if t_label == prediction:
            for j in range(len(thresh_list)):
                if radius > thresh_list[j]:
                    correct_list[j] += 1
        n_total += 1

    for i in range(len(thresh_list)):
        print("sigmoid: %.1f, radius: %.1f, acc: %.3f" % (args.sigma, thresh_list[i], correct_list[i]*1.0 / n_total))







