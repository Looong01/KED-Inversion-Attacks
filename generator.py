import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    """
    Here is the comments for the code above:
    1. `self.diffusion` is a list of modules, each of which is an nn.Sequential module. Each of these modules first performs a convolution and then applies a ReLU activation function. The output of this module is the diffusion process, which is then applied to the encoded input `z`.
    2. The diffusion process is performed for `num_steps` steps. For each step, a noise tensor is sampled from a normal distribution and added to the encoded input `z`. This noise tensor is then multiplied by the time step `t[step]`, which is a tensor of shape `(batch_size,)` containing the values `0.0` or `1.0`. This effectively multiplies the noise tensor by `1.0` when `t[step]` is `0.0` and by `0.0` when `t[step]` is `1.0`. This means that the noise tensor is only added to `z` when `t[step]` is `0.0`, which corresponds to the first time step of the diffusion process. When `t[step]` is `1.0`, the noise tensor is not added to `z`, meaning that the encoded input is not modified. This process is repeated for each step of the diffusion process. This is the key step of the diffusion model, which makes it different from the VAE.
    3. The final encoded input `z` is decoded into the output `x_hat`.
    """
    def __init__(self, in_channels, out_channels, dim, num_steps):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Define the diffusion process
        self.diffusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_steps)
        ])

        # Define the decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Encode the input
        z = self.encoder(x)

        # Perform the diffusion process
        for step in range(self.num_steps):
            noise = torch.randn_like(x)
            z = z + noise * (1.0 - t[step])
            z = self.diffusion[step](z)

        # Decode the final state
        x_hat = self.decoder(z)

        return x_hat

class Generator(nn.Module):
    '''
    Rewrite the generator to use the diffusion model
    '''
    def __init__(self, in_dim=100, dim=128, num_steps=100):
        super(Generator, self).__init__()
        self.num_steps = num_steps

        # Define the initial linear layer
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.LeakyReLU(0.15)
        )

        # Define the diffusion model
        self.diffusion = DiffusionModel(in_channels=dim, out_channels=3, dim=dim, num_steps=num_steps)

    def forward(self, x, t):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y_hat = self.diffusion(y, t)
        return y_hat

# class Generator(nn.Module):
#     def __init__(self, in_dim=100, dim=128):
#         super(Generator, self).__init__()
#         def dconv_bn_relu(in_dim, out_dim):
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
#                                    padding=2, output_padding=1, bias=False),
#                 nn.BatchNorm2d(out_dim),
#                 nn.LeakyReLU(0.15))

#         self.l1 = nn.Sequential(
#             nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
#             nn.BatchNorm1d(dim * 8 * 4 * 4),
#             nn.LeakyReLU(0.15))
#         self.l2_5 = nn.Sequential(
#             dconv_bn_relu(dim * 8, dim * 4),
#             dconv_bn_relu(dim * 4, dim * 2),
#             dconv_bn_relu(dim * 2, dim * 2),
#             dconv_bn_relu(dim * 2, dim * 2),
#             dconv_bn_relu(dim * 2, dim),
#             nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
#             nn.Sigmoid())

#     def forward(self, x):
#         y = self.l1(x)
#         y = y.view(y.size(0), -1, 4, 4)
#         y = self.l2_5(y)
#         return y

class GeneratorMNIST(nn.Module):
    '''
    Generator for MNIST dataset
    '''
    def __init__(self, in_dim=100, dim=64):
        super(GeneratorMNIST, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 4 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 4 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 1, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class CompletionNetwork(nn.Module):
    '''
    Completion Network
    '''
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(128)
        self.act7 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(128)
        self.act8 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(128)
        self.act9 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(128)
        self.act10 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(128)
        self.act11 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        self.act12 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(64)
        self.act13 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.act14 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(32)
        self.act15 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()
        # output_shape: (None, 3, img_h. img_w)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x

def dconv_bn_relu(in_dim, out_dim):
    '''
    Deconvolution + BatchNormalization + ReLU
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())

class ContextNetwork(nn.Module):
    '''
    Context Network
    '''
    def __init__(self):
        super(ContextNetwork, self).__init__()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(128)
        self.act7 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(128)
        self.act8 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(128)
        self.act9 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(128)
        self.act10 = nn.ReLU()
        
        

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        return x

class IdentityGenerator(nn.Module):

    def __init__(self, in_dim = 100, dim=64):
        super(IdentityGenerator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2))

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class InversionNet(nn.Module):
    '''
    Inversion Network
    '''
    def __init__(self, out_dim = 128):
        super(InversionNet, self).__init__()
        
        # input [4, h, w]  output [256, h // 4, w // 4]
        self.ContextNetwork = ContextNetwork()
        # input [z_dim] output[128, 16, 16]
        self.IdentityGenerator = IdentityGenerator()

        self.dim = 128 + 128
        self.out_dim = out_dim
        
        self.Dconv = nn.Sequential(
            dconv_bn_relu(self.dim, self.out_dim),
            dconv_bn_relu(self.out_dim, self.out_dim // 2))

        self.Conv = nn.Sequential(
            nn.Conv2d(self.out_dim // 2, self.out_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim // 4),
            nn.ReLU(),
            nn.Conv2d(self.out_dim // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())


    def forward(self, inp):
        # x.shape [4, h, w]  z.shape [100]
        x, z = inp
        context_info = self.ContextNetwork(x)
        identity_info = self.IdentityGenerator(z)
        y = torch.cat((context_info, identity_info), dim=1)
        y = self.Dconv(y)
        y = self.Conv(y)

        return y
