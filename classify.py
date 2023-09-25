# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
from torchvision.models import VGG16_BN_Weights
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math, evolve


class Flatten(nn.Module):
    '''
    Flatten the input
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)


class Mnist_CNN(nn.Module):
    """
    Here is the comments for the code above:
    1. First, we define a class called Mnist_CNN which is a child class of nn.Module. In the child class, we define the __init__ function and the forward function.
    2. In the __init__ function, we define the structure of the CNN. The structure is exactly the same as the structure of the CNN we used in the last lab.
    3. In the forward function, we define the forward process of the CNN. The forward process is also exactly the same as the forward process of the CNN we used in the last lab.
    4. In the last line, we return the output of the CNN. 
    """
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        res = self.fc2(x)
        return [x, res]


class VGG16(nn.Module):
    """
    Here is the comments for the code above:
    1. model.features is a Sequential object containing all the convolutional layers of the network. By using this object as a function we can use it to extract the feature map from an image tensor.
    2. model.classifier is also a Sequential object containing all the fully connected layers of the original network. The last layer of this object is a Linear layer with 1000 output features corresponding to the 1000 ImageNet's classes. We can replace this last layer with another Linear layer with 2 output features, corresponding to our classes.
    3. The forward method is the one called when we pass an image tensor to the model object. The output of this method is the output of the last layer of the network, i.e. the logits. We can also define a predict method to get the actual class prediction from the logits.
    """
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return [feature, res]

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out


class VGG16_vib(nn.Module):
    """
    Same as VGG16, but with variational inference
    """
    def __init__(self, n_classes):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.k = self.feat_dim // 2
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)

    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return [feature, out, mu, std]

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return out


class CrossEntropyLoss(_Loss):
    '''
    Cross entropy loss
    '''
    def forward(self, out, gt, mode="reg"):
        bs = out.size(0)
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        if mode == "dp":
            loss = torch.sum(loss, dim=1).view(-1)
        else:
            loss = torch.sum(loss) / bs
        return loss


class BinaryLoss(_Loss):
    '''
    Binary loss
    '''
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - (gt * torch.log(out.float() + 1e-7) + (1 - gt) * torch.log(1 - out.float() + 1e-7))
        loss = torch.mean(loss)
        return loss


class FaceNet(nn.Module):
    """
    Here is the comments for the code above:
    1. The IR_50_112 is the network we used in this project. It is a pre-trained model, so we can take it as a "black box" and use it only when we need.
    2. The forward function is the function which will be called by the program when we run the program.
    3. The predict function is the function which will be called when we want to do prediction after we train the model.
    """
    def __init__(self, num_classes=1000):
        super(FaceNet, self).__init__()
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def predict(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out

    def forward(self, x):
        # print("input shape:", x.shape)
        # import pdb; pdb.set_trace()

        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return [feat, out]


class FaceNet64(nn.Module):
    """
    Here is the comments for the code above:
    1. The input images are resized to (64, 64), and the batch size is set to 64.
    2. The architecture of the model is IR_50_64, which is a ResNet-50 with the input size of 64x64.
    3. The output of the ResNet-50 is a 512-dim feature vector.
    4. The output of the model is a 512-dim feature vector and a 1000-dim fc layer.
    5. The feature vector is used for clustering, and the fc layer is used for classification. 
    """
    def __init__(self, num_classes=1000):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out


class IR152(nn.Module):
    """
    Here is the comments for the code above:
    1. The model is inherited from the nn.Module class, which is the base class for all neural network modules.
    2. The __init__() function is used to initialize the model. In the function, we first define a feature extractor and an output layer. The feature extractor is a modified version of the ResNet-152, which is trained with the ArcFace loss. The output layer consists of a batch normalization layer, a dropout layer, a flatten layer, a linear layer and a batch normalization layer.
    3. The forward() function defines how the input is processed in the model. The input is first fed into the feature extractor, and then the output of the feature extractor is fed into the output layer to obtain the feature embedding and the classification result.
    4. In the feature extractor, we use the Flatten() function to flatten the output of the convolutional layer into a one-dimensional vector. This is because the input of the linear layer should be a one-dimensional vector. The output of the feature extractor is a 512-dimensional vector, which is the feature embedding of the input image.
    5. In the output layer, we use the nn.Linear() function to define the linear layer, and use the nn.BatchNorm1d() function to define the batch normalization layer. The output of the linear layer is a 1000-dimensional vector, which is the prediction of the classification.
    """
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out


class IR152_vib(nn.Module):
    '''
    Same as IR152, but with variational inference
    '''
    def __init__(self, num_classes=1000):
        super(IR152_vib, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.n_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, std


class IR50(nn.Module):
    """
    Here is the comments for the code above:
    1. The model is Resnet50 + BN + Dropout + Flatten + FC(512)
    2. The output of the FC layer is the feature of the images
    3. The st_layer is a linear layer to predict the mean and std of the Gaussian distribution
    4. The fc_layer is a linear layer to predict the labels
    5. The mu and std are the parameters of the Gaussian distribution
    6. The eps is a random noise sampled from a Gaussian distribution
    7. The res is the reparameterization trick of the Gaussian distribution
    8. The iden is the predicted labels
    """
    def __init__(self, num_classes=1000):
        super(IR50, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, std


class IR50_vib(nn.Module):
    '''
    Same as IR50, but with variational inference
    '''
    def __init__(self, num_classes=1000):
        super(IR50_vib, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.n_classes = num_classes
        self.k = self.feat_dim // 2
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feat = self.output_layer(self.feature(x))
        feat = feat.view(feat.size(0), -1)
        statis = self.st_layer(feat)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feat, out, iden, mu, std
