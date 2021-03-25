# evaluate a smoothed classifier on a dataset
import argparse
import os
# from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from models import CNN
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
# from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default="mnist", help="which dataset")
parser.add_argument("--base_classifier", default="outputs/garbage1/best_model.pt", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", default="0.5", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", default="certification.txt", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


#python code/certify.py imagenet model_output_dir/checkpoint.pth.tar 0.50 certification_output
# --alpha 0.001 --N0 100 --N 100000 --skip 100 --batch 400

# r            0      0.5     1      1.5      2     2.5    3.0
#sigma  0.25   
#       0.5    0.683  0.402   0.191  0.046    0      0      0
#        1



if __name__ == "__main__":
    # load the base classifier
    device = 'cuda:0'

    target_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    target_dataset_test = MNIST(
        './input', train=False, transform=target_transform, download=True)

    target_test_loader = DataLoader(
        target_dataset_test, batch_size=1, shuffle=False,
        num_workers=1)

    checkpoint = torch.load(args.base_classifier)
    base_classifier = CNN(in_channels=3, target=True).to(device)
    base_classifier.load_state_dict(checkpoint['model'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, num_classes= 10, sigma=args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    n_total = 0
    n_correct = 0
    thresh_list = [0,0.5,1.0,1.5,2.0,2.5,3.0]
    correct_list = [0]*7
    for i,data in enumerate(target_test_loader):
        if i % 100 == 0:
            print(i)
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = data

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        label = label.item()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        if label == prediction:
            for j in range(len(thresh_list)):
                if radius > thresh_list[j]:
                    correct_list[j] += 1
        n_total += 1
    f.close()

    for i in range(len(thresh_list)):
        print("sigmoid: %.1f, radius: %.1f, acc: %.3f" % (args.sigma, thresh_list[i], correct_list[i]*1.0 / n_total))



