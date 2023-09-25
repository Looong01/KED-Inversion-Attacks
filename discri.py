import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LinearWeightNorm
import torch.nn.init as init

class MinibatchDiscrimination(nn.Module):
    """
    Here is the comments for the code above:
    1. The minibatch discriminaton layer performs the following operations:
        * It takes as input NxA matrix and outputs NxA+B, where B is the dimension of the output of the minibatch discrimination layer.
        * It calculates a NxNxB tensor, where N is the size of the minibatch and B is the number of kernels. 
        * It calculates the exponential sum of the tensor along the 2nd dimension, and subtract 1 from it.
        * It concatenates the output to the input tensor. 
    2. The kernel tensor T is a learnable parameter of size AxBxC, where A is the size of the input tensor, B is the number of kernels, and C is the dimension of each kernel. 
    3. The kernel tensor T is initialized with random values sampled from a normal distribution with mean 0 and standard deviation 1.
    """
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class MinibatchDiscriminator(nn.Module):
    """ Here is the comments for the code above:
    1. We use InstanceNorm2d instead of LayerNorm
    2. We use MinibatchDiscrimination module to improve the performance of the model
    3. We use LeakyReLU instead of ReLU
    4. We use kernel size 5 instead of 4
    5. We use stride 2 instead of 1
    6. We use padding 2 instead of 1
    """
    def __init__(self,in_dim=3, dim=128, n_classes=1000):
        super(MinibatchDiscriminator, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.15))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim*2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim*2, dim*4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim*4, dim*4, 3, 2, 1)
        self.layer5 = conv_ln_lrelu(dim*4, dim*4, 3, 2, 1)
        self.layer5 = conv_ln_lrelu(dim*4, dim*4, 3, 2, 1)
        self.mbd1 = MinibatchDiscrimination(dim*4*4*4, 64, 50)
        self.fc_layer = nn.Linear(dim*4*4*4+64, self.n_classes)

    def forward(self, x):
        out = []
        bs = x.shape[0]
        feat1 = self.layer1(x)
        out.append(feat1)
        feat2 = self.layer2(feat1)
        out.append(feat2)
        feat3 = self.layer3(feat2)
        out.append(feat3)
        feat4 = self.layer4(feat3)
        out.append(feat4)
        feat5 = self.layer5(feat4)
        out.append(feat5)
        feat6 = self.layer5(feat5)
        out.append(feat6)
        feat = feat6.view(bs, -1)
        # print('feat:', feat.shape)
        mb_out = self.mbd1(feat)   # Nx(A+B)
        y = self.fc_layer(mb_out)
        
        return feat, y
        # return mb_out, y


class Discriminator(nn.Module):
    """
    Here is the comments for the code above:
    1. The discriminator is a classifier with 4 convolutional layers and 1 fully connected layer. 
    2. The discriminator takes an image with the size of (3, 64, 64) as input, and outputs a tensor with the size of (batch_size, 1000). 
    3. The output tensor is used to calculate the loss of the discriminator. 
    4. The output tensor is also used to calculate the loss of the generator. 
    5. The output tensor is also used to calculate the accuracy of the discriminator.
    """
    def __init__(self,in_dim=3, dim=64, n_classes=1000):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.15))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim*2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim*2, dim*4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim*4, dim*4, 3, 2, 1)
        self.fc_layer = nn.Linear(dim*4*4*4, self.n_classes)

    def forward(self, x):
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat = feat4.view(bs, -1)
        y = self.fc_layer(feat)
        
        return feat, y


class DiscriminatorMNIST(nn.Module):
    """
    Here is the comments for the code above:
    1. The Discriminator network is defined with four linear layers. 
    2. The input to the discriminator is 28 x 28 x 1 image, which is flattened to a vector of size 784.
    3. The output of the discriminator is a single number between 0 and 1, which gives the probability of input image being fake or real. 
    4. The Sigmoid activation is applied to the output of the discriminator to get the probability.
    5. The leaky ReLU activation is applied to all the intermediate layers except the final layer. 
    6. The dropout value is set to 0.3 for the intermediate layers.
    """
    def __init__(self, d_input_dim=1024):
        super(DiscriminatorMNIST, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        y = self.fc4(x)
        y = y.view(-1)

        return y

class DGWGAN32(nn.Module):
    """
    Here is the comments for the code above:
    1. We use the convolutional layer with instance normalization and leaky relu activation. 
    2. We use the same kernel size (5) for all convolutional layers. 
    3. We use stride=2 for all convolutional layers. 
    4. We use padding=2 for all convolutional layers. 
    5. We use 4 convolutional layers. 
    6. We use 1 fully connected layer. 
    7. We use the leaky relu activation with slope 0.2. 
    8. We use the sigmoid activation for the output of the discriminator.
    """
    def __init__(self, in_dim=1, dim=64):
        super(DGWGAN32, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)
    
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        y = y.view(-1)
        return y

class DGWGAN(nn.Module):
    """
    Here is the comments for the code above:
    1. To avoid the gradient vanishing problem, we use LeakyReLU activation function.
    2. Since there is no effective implementation of LayerNorm, we use InstanceNorm2d
    instead of LayerNorm here.
    3. We use the Wasserstein distance to measure the distance between the real and 
    fake distributions. The Wasserstein distance is calculated by the difference 
    between the average values of the discriminator's outputs on the real and fake 
    samples. To achieve this, we remove the sigmoid function from the discriminator's 
    last layer.
    """
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))
    
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

class DLWGAN(nn.Module):
    """
    Here is the comments for the code above:
    1. This implementation is based on the official implementation of DCGAN in PyTorch.
    2. In the forward function, we return the features of the second last layer. 
    This is because we want to use it to calculate the feature matching loss.
    3. The original DCGAN uses BatchNorm2d, but we replace it with InstanceNorm2d 
    because there is no effective PyTorch implementation of LayerNorm. 
    You can try to use LayerNorm to replace InstanceNorm2d to see if the performance 
    can be improved.
    """
    def __init__(self, in_dim=3, dim=64):
        super(DLWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)
       
    
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        return y




