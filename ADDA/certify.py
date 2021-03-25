# evaluate a smoothed classifier on a dataset
import numpy as np
np.set_printoptions(threshold=100,precision=4,suppress=True)
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICE'] = '2'
# from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from models import CNN
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import GetLoader
from data_load import svhn, usps
# from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default="mnist", help="which dataset")
parser.add_argument("--base_classifier", default="outputs/garbage1/best_model.pt", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", default="1.0", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", default="certification.txt", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


#python code/certify.py imagenet model_output_dir/checkpoint.pth.tar 0.50 certification_output
# --alpha 0.001 --N0 100 --N 100000 --skip 100 --batch 400

# MNIST_MM
# r            0      0.5     1      1.5      2     2.5    3.0
#sigma  0.25   0.527  0.346  0.0     0.0     0.0    0.0    0.0
#       0.5    0.277  0.183  0.130  0.075    0      0      0
#        1     0.142  0.122  0.108  0.091    0.062  0.037  0.009

# usps
# r            0      0.5     1      1.5      2     2.5    3.0
#sigma  0.25   0.221  0.108  0.0     0.0     0.0    0.0    0.0
#       0.5    0.167  0.104  0.035  0.004    0      0      0
#        1     0.146  0.108  0.050  0.016    0.004  0.0    0.0

if __name__ == "__main__":
    # load the base classifier
    device = 'cuda:0'
    batch_size = 1

    # target_dataset_name = 'usps'
    target_dataset_name = 'mnist_m'
    print(target_dataset_name,args.sigma)
    if target_dataset_name=='mnist_m':
        args.base_classifier = "outputs/garbage1_m2mm/best_model.pt"
        target_image_root = os.path.join('dataset', target_dataset_name)
        image_size = 28


        img_transform_target = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        test_list = os.path.join(target_image_root, 'mnist_m_test_labels.txt')

        dataset_target_test = GetLoader(
            data_root=os.path.join(target_image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    elif target_dataset_name=='usps':
        args.base_classifier = "outputs/garbage1_s2u/best_model.pt"
        dataset_target_test = usps.USPS('./data/usps/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))
                                        ]))


    target_test_loader = torch.utils.data.DataLoader(
        dataset=dataset_target_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )


    checkpoint = torch.load(args.base_classifier)
    base_classifier = CNN(in_channels=3, target=True).to(device)
    base_classifier.load_state_dict(checkpoint['model'])

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
        # # only certify every args.skip examples, and stop after args.max examples
        # if i % args.skip != 0:
        #     continue
        # if i == args.max:
        #     break
        (x, label) = data
        save_file = open('output/'+ target_dataset_name + "_" + str(args.sigma)+ "_"+ str(i) + "_"+ str(label.item()) + "_softmax.txt","w")

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch,save_file)
        after_time = time()
        label = label.item()
        correct = int(prediction == label)
        print("index:",i,"label:",label,"prediction:",prediction,"radius:",radius,file=save_file)

        # time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        # print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            # i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        if label == prediction:
            for j in range(len(thresh_list)):
                if radius > thresh_list[j]:
                    correct_list[j] += 1
        n_total += 1
    # f.close()

    for i in range(len(thresh_list)):
        print("sigmoid: %.3f, radius: %.1f, acc: %.3f" % (args.sigma, thresh_list[i], correct_list[i]*1.0 / n_total))



