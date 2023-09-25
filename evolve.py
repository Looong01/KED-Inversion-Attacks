import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
 
 
# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
 
 
class Flatten(Module):
    '''
    Flattens the input. Does not affect the batch size.
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)
 
 
def l2_norm(input, axis=1):
    '''
    Performs L2 normalization operation on the input tensor.
    Does not affect the batch size.
    '''
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
 
    return output
 
 
class SEModule(Module):
    """
    Here is the comments for the code above:
    1. The first part of the __init__ function is to call the __init__ function of the parent class Module.
    2. The class AdaptiveAvgPool2d is a function that can automatically calculate the kernel_size and stride according to the input size.
    3. The class Conv2d is a function that can automatically calculate the input and output channels according to the input.
    4. Then we define the forward function. Here, we first assign the input to module_input. Then we pass the input to the avg_pool layer and then to the fc1 layer. Next, we pass the output of fc1 to the relu layer and then to the fc2 layer. Finally, we pass the output of fc2 to the sigmoid layer and multiply it with the module_input. The output of this module is the product of the input and the output of sigmoid.
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
 
        nn.init.xavier_uniform_(self.fc1.weight.data)
 
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
 
        self.sigmoid = Sigmoid()
 
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
 
        return module_input * x
 
 
class bottleneck_IR(Module):
    """
    Here is the comments for the code above:
    1. the first if-else statement is to deal with the situation that the input channel and output channel are the same.
    2. the shortcut layer is a sequence of maxpooling and conv2d.
    3. the res_layer is a sequence of batchnorm, conv2d, prelu, conv2d, batchnorm.
    4. the forward function is to add the shortcut and the residual layers.
    """
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))
 
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
 
        return res + shortcut
 
 
class bottleneck_IR_SE(Module):
    """
    Here is the comments for the code above:
    1. The bottleneck_IR_SE is a class that inherits from torch.nn.Module. It is defined as a block with a residual structure.
    2. The bottleneck_IR_SE has three main parts: a shortcut layer, a residual layer and a SE layer.
    3. The shortcut layer is used to match the input and output dimensions. We can use MaxPool2d to reduce the input feature map size and the number of channels.
    4. The residual layer is used to extract features. It contains two convolutional layers and a PReLU activation function. The first convolutional layer is used to extract the feature of the input. The second convolutional layer is used to fuse the features extracted by the first convolutional layer and the features of the shortcut layer.
    5. The SE layer is used to recalibrate the channel-wise feature responses by explicitly modeling interdependencies between channels.
    """
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )
 
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
 
        return res + shortcut
 
 
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
 
 
def get_block(in_channel, depth, num_units, stride=2):
    '''
    Creates a bottleneck block.
    '''
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]
 
 
def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
 
    return blocks
 
 
class Backbone64(Module):
    """
    Here is the comments for the code above:
    1. The model has two parts: input_layer and body. The input_layer contains three layers: Conv2d, BatchNorm2d and PReLU; The body contains many blocks, each block contains many bottleneck units. The original paper use different block for different number of layers.
    2. The bottleneck_IR and bottleneck_IR_SE are both inherit from nn.Module, and the difference between them is bottleneck_IR_SE has a SE layer.
    3. The forward function of the model is to forward the input into input_layer and then into body.
    4. The _initialize_weights function is to initialize the weight of each layer, and the parameters of each layer is stored in self.modules().
    """
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone64, self).__init__()
        assert input_size[0] in [64, 112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
 
        self._initialize_weights()
 
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        #x = self.output_layer(x)
 
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class Backbone112(Module):
    """
    Here is the comments for the code above:
    1. We can see that, the basic unit of the backbone is defined in the bottleneck_IR(SE) class.
    2. The backbone is defined in the Backbone112 class. The input size and the number of layers are the parameters of this class.
    3. The input layer is a convolution layer followed by a batch normalization layer and a PReLU layer.
    4. The output layer is a batch normalization layer followed by a dropout layer, a flatten layer, a linear layer and a batch normalization layer.
    5. The body is defined by the basic unit and the number of layers.
    6. The _initialize_weights function is used to initialize the parameters.
    7. The forward function is used to forward the input and get the output.
    8. The output of the backbone is a 512-dimensional feature vector.
    """
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone112, self).__init__()
        assert input_size[0] in [64, 112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))
 
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
 
        self._initialize_weights()
 
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
 
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
 
 
def IR_50_64(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone64(input_size, 50, 'ir')
 
    return model

def IR_50_112(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone112(input_size, 50, 'ir')
 
    return model
 
 
def IR_100(input_size):
    """Constructs a ir-100 model.
    """
    model = Backbone64(input_size, 100, 'ir')
 
    return model

def IR_152_64(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone64(input_size, 152, 'ir')
 
    return model
 
 
def IR_152_112(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone112(input_size, 152, 'ir')
 
    return model

def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone64(input_size, 50, 'ir_se')
 
    return model
 
 
def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone112(input_size, 100, 'ir_se')
 
    return model
 
 
def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone112(input_size, 152, 'ir_se')
 
    return model