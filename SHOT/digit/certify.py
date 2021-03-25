# evaluate a smoothed classifier on a dataset
import numpy as np
np.set_printoptions(threshold=100,precision=4,suppress=True)
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# from datasets import get_dataset, DATASETS, get_num_classes
from digit.core import Smooth
from time import time
import torch
import datetime
import os.path as osp
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
# from architectures import get_architecture
from digit import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from digit.data_load import mnist, svhn, usps
from digit.uda_digit_s2u import digit_load

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default="mnist", help="which dataset")
parser.add_argument("--base_classifier", default="outputs/garbage1/best_model.pt", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", default=1.0, type=float, help="noise hyperparameter")
parser.add_argument("--outfile", default="certification.txt", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--s', type=int, default=0, help="source")
parser.add_argument('--t', type=int, default=1, help="target")
parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
parser.add_argument('--worker', type=int, default=4, help="number of workers")
parser.add_argument('--dset', type=str, default='m2mm', choices=['u2m', 'm2u','s2m','m2mm'])
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--seed', type=int, default=2020, help="random seed")
parser.add_argument('--cls_par', type=float, default=0.3)
parser.add_argument('--ent_par', type=float, default=1.0)
parser.add_argument('--gent', type=bool, default=True)
parser.add_argument('--ent', type=bool, default=True)
parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
parser.add_argument('--smooth', type=float, default=0.1)
parser.add_argument('--output', type=str, default='')
parser.add_argument('--issave', type=bool, default=True)

args = parser.parse_args()
args.class_num = 10
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
#python code/certify.py imagenet model_output_dir/checkpoint.pth.tar 0.50 certification_output
# --alpha 0.001 --N0 100 --N 100000 --skip 100 --batch 400

# r            0      0.5     1      1.5      2     2.5    3.0
#sigma  0.25   0.403  0.265   0       0       0       0     0
#       0.5    0.274  0.213   0.138   0       0       0      0
#        1     0.185  0.148   0.097   0.052   0.015   0      0


# r            0      0.5     1      1.5      2     2.5    3.0
#sigma  0.25   0.932  0.842   0       0       0      0      0
#       0.5    0.808  0.538   0.245   0       0      0      0
#        1     0.262  0.122   0.049   0.020   0.002  0.000  0.000


if __name__ == "__main__":
    # load the base classifier
    device = 'cuda:0'
    args.dset = 'm2mm'
    # args.dset = 's2u'
    print(args.dset, args.sigma)
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset == 'm2mm':
        netF = network.DTNBase_c().cuda()
    elif args.dset == 's2u':
        netF = network.DTNBase_c().cuda()
    print(args.dset)
    netB = network.feat_bootleneck_c().cuda()
    netC = network.feat_classifier_c().cuda()

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    args.modelpath = args.output_dir + '/target_F_par_0.3.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_B_par_0.3.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/target_C_par_0.3.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()
    target_test_loader = digit_load(args)['test']

    base_classifier = {}
    base_classifier["netF"] = netF
    base_classifier["netB"] = netB
    base_classifier["netC"] = netC


    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, num_classes= 10, sigma=args.sigma)

    # prepare output file
    # f = open(args.outfile, 'w')
    # print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    n_total = 0
    n_correct = 0
    thresh_list = [0,0.5,1.0,1.5,2.0,2.5,3.0]
    correct_list = [0]*7
    for i,data in enumerate(target_test_loader):
        if i % 100 == 0:
            print(i)
        # if i > 100:
        #     break
        # only certify every args.skip examples, and stop after args.max examples
        # if i % args.skip != 0:
        #     continue
        # if i == args.max:
        #     break

        (x, label) = data
        save_file = open('output/'+ args.dset + "_" + str(args.sigma)+ "_"+ str(i) + "_"+ str(label.item()) + "_softmax.txt","w")
        print
        # before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, save_file)
        after_time = time()
        correct = int(prediction == label)
        print("index:",i,"label:",label.item(),"prediction:",prediction,"radius:",radius,file=save_file)
        # time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        # print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        #     i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        if label == prediction:
            for j in range(len(thresh_list)):
                if radius > thresh_list[j]:
                    correct_list[j] += 1
        n_total += 1
    # f.close()

    for i in range(len(thresh_list)):
        print("sigmoid: %.2f, radius: %.1f, acc: %.3f" % (args.sigma, thresh_list[i], correct_list[i]*1.0 / n_total))



