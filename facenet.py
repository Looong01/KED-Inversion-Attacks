import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
 
 
# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
class FaceNet(nn.Module):
    '''
    Define the FaceNet model
    '''
    def __init__(self, num_classes = 1000):
        super(FaceNet, self).__init__()
        self.feature = IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return feat, out, iden

class FaceNet64(nn.Module):
    '''
    Same as FaceNet but with input size 64x64
    '''
    def __init__(self, num_classes = 1000):
        super(FaceNet64, self).__init__()
        self.feature = IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return feat, out, iden

class Flatten(Module):
    '''
    Flatten the input
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)
 
 
def l2_norm(input, axis=1):
    '''
    Perform l2 normalization operation on the input vector
    '''
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
 
    return output
 
 
class SEModule(Module):
    '''
    Squeeze-and-Excitation Module
    '''
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
    '''
    bottleneck residual block for IR models
    '''
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
    '''
    Same as bottleneck_IR, but with Squeeze-and-Excitation module
    '''
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
    '''
    Creates the block units for the corresponding ResNet model.
    '''
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
    else:
        return None
    return blocks
 
 
class Backbone64(Module):
    '''
    Define the backbone network
    '''
    def __init__(self, input_size, num_layers, mode='ir'):
        global unit_module
        super(Backbone64, self).__init__()
        assert input_size[0] in [64], "input_size should be [112, 112] or [224, 224]"
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
    '''
    Same as Backbone64 but with input size 112x112
    '''
    def __init__(self, input_size, num_layers, mode='ir'):
        global unit_module
        super(Backbone112, self).__init__()
        assert input_size[0] in [112], "input_size should be [112, 112] or [224, 224]"
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
 
 
def IR_101(input_size):
    """Constructs a ir-101 model.
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
    model = Backbone112(input_size, 50, 'ir_se')
 
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