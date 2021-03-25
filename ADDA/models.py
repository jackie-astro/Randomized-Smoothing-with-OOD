from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))


    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        return feature


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, feature):
        class_output = self.class_classifier(feature)
        return class_output






class CNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, target=False):
        super(CNN, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, h=500, args=None):
        super(Discriminator, self).__init__()

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # self.l1 = nn.Linear(500, h)
        # self.l2 = nn.Linear(h, h)
        # self.l3 = nn.Linear(h, 2)
        # self.slope = args.slope
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # x = F.leaky_relu(self.l1(x), self.slope)
        # x = F.leaky_relu(self.l2(x), self.slope)
        # x = self.l3(x)
        x = self.domain_classifier(x)

        return x
