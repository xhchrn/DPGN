import torch
import torch.nn as nn
import torch.nn.functional as F
from film import FiLM_Layer
from dual_bn import DualBN1d, DualBN2d


class ResNet12Block_FiLM(nn.Module):
    """
    ResNet block
    """
    def __init__(self, inplanes, planes,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                 film_normalize=False, dual_BN=True):
        super(ResNet12Block_FiLM, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.film1 = FiLM_Layer(planes,
                                in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act,
                                normalize=film_normalize)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.film2 = FiLM_Layer(planes,
                                in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act,
                                normalize=film_normalize)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.film3 = FiLM_Layer(planes,
                                in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act,
                                normalize=film_normalize)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = DualBN2d(planes) if dual_BN else nn.BatchNorm2d(planes)
        self.film = FiLM_Layer(planes,
                               in_channels=film_indim,
                               alpha=film_alpha,
                               activation=film_act,
                               normalize=film_normalize)
        # self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dual_BN = dual_BN

    def forward(self, x, task_embedding, n_expand):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual, task_embedding) if self.dual_BN else self.bn(residual)
        residual = self.film(residual, task_embedding, n_expand)

        out = self.conv1(x)
        out = self.bn1(out, task_embedding) if self.dual_BN else self.bn1(out)
        out = self.film1(out, task_embedding, n_expand)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, task_embedding) if self.dual_BN else self.bn2(out)
        out = self.film2(out, task_embedding, n_expand)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, task_embedding) if self.dual_BN else self.bn3(out)
        # out = self.bn3(out)
        out = self.film3(out, task_embedding, n_expand)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class ResNet18Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, downsample=False):

        super(ResNet18Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.conv(residual)
            residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)
        if self.downsample:
            out = self.maxpool(out)

        return out


class WRNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WRNBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet12_FiLM(nn.Module):
    """
    ResNet12 backbone
    """
    def __init__(self, emb_size, block=ResNet12Block_FiLM, cifar_flag=False,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                 film_normalize=False, dual_BN=True):
        super(ResNet12_FiLM, self).__init__()
        cfg = [64, 128, 256, 512]
        # layers = [1, 1, 1, 1]
        iChannels = int(cfg[0])
        self.dual_BN = dual_BN
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = DualBN2d(iChannels) if dual_BN else nn.BatchNorm2d(iChannels)
        # self.bn1 = nn.BatchNorm2d(iChannels)
        self.film1 = FiLM_Layer(iChannels,
                                in_channels=film_indim,
                                alpha=film_alpha,
                                activation=film_act,
                                normalize=film_normalize)
        self.relu = nn.LeakyReLU()

        self.emb_size = emb_size
        self.layer1 = self._make_layer(
            block, cfg[0], cfg[0],
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN
        )
        self.layer2 = self._make_layer(
            block, cfg[0], cfg[1],
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN
        )
        self.layer3 = self._make_layer(
            block, cfg[1], cfg[2],
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN
        )
        self.layer4 = self._make_layer(
            block, cfg[2], cfg[3],
            film_indim=film_indim, film_alpha=film_alpha, film_act=film_act,
            film_normalize=film_normalize, dual_BN=dual_BN
        )
        self.avgpool = nn.AvgPool2d(7)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        self.layer_second = nn.Sequential(nn.Linear(in_features=layer_second_in_feat,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        # self.layer_second_fc = nn.Linear(in_features=layer_second_in_feat,
        #                                  out_features=self.emb_size,
        #                                  bias=True)
        # self.layer_second_bn = DualBN1d(self.emb_size) if dual_BN else nn.BatchNorm1d(self.emb_size)
        # self.layer_second_film = FiLM_Layer()

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes,
                    film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                    film_normalize=False, dual_BN=True):
        # layers = []
        # layers.append(block(inplanes, planes))
        layers = block(inplanes, planes, film_indim, film_alpha, film_act,
                       film_normalize, dual_BN)
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x, task_embedding, n_expand):
        # 3 -> 64
        x = self.conv1(x)
        x = self.bn1(x, task_embedding) if self.dual_BN else self.bn1(x)
        # x = self.bn1(x)
        x = self.film1(x, task_embedding, n_expand)
        x = self.relu(x)
        # 64 -> 64
        x = self.layer1(x, task_embedding, n_expand)
        # 64 -> 128
        x = self.layer2(x, task_embedding, n_expand)
        # 128 -> 256
        inter = self.layer3(x, task_embedding, n_expand)
        # 256 -> 512
        x = self.layer4(inter, task_embedding, n_expand)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # 512 -> 128
        x = self.layer_last(x)
        inter = self.maxpool(inter)
        # 256 * 5 * 5
        inter = inter.view(inter.size(0), -1)
        # 256 * 5 * 5 -> 128
        inter = self.layer_second(inter)
        out = []
        out.append(x)
        out.append(inter)
        # no FC here
        return out


def get_task_embedding_func():
    from task_embedding import TaskEmbedding
    # Choose the task embedding function
    te_func = TaskEmbedding(metric='FiLM_SVM_WGrad')

    # device_ids = list(range(len(options.gpu.split(','))))
    # te_func = torch.nn.DataParallel(te_func, device_ids=device_ids)

    return te_func


class ResNet12_FiLM_Encoder(nn.Module):
    """
    ResNet12 backbone with FiLM layers
    """
    def __init__(self, emb_size, block=ResNet12Block_FiLM, cifar_flag=False,
                 film_indim=1, film_alpha=1, film_act=F.leaky_relu,
                 film_normalize=False, dual_BN=True):
        super(ResNet12_FiLM_Encoder, self).__init__()
        self.resnet12 = ResNet12_FiLM(emb_size, block, cifar_flag,
                                      film_indim, film_alpha, film_act,
                                      film_normalize, dual_BN)
        self.add_te_func = get_task_embedding_func()

    def forward(self, x, support_label):
        # NOTE: backbone only take samples in one task at a time;
        #   refer to `utils.backbone_two_stage_initialization`
        assert x.size(0) == 1 and support_label.size(0) == 1
        x = x.squeeze(0)
        support_label = support_label.squeeze(0)
        if x.size(0) == 30:
            n_way, n_shot, n_support = 5, 5, 25
        elif x.size(0) == 10:
            n_way, n_shot, n_support = 5, 1, 5
        else:
            print(x.size())
            raise ValueError('current configuration not implemented')
        assert support_label.size(0) == n_support

        support_data = x[:n_support]
        query_data = x[n_support:]

        # first pass without task embedding
        # with torch.no_grad():
        support_encode = self.resnet12(support_data, None, None)
        support_encode = torch.cat(support_encode, dim=1)
        # support_encode = support_encode.detach().requires_grad_()
        emb_task, _ = self.add_te_func(
            support_encode.unsqueeze(0), support_label.unsqueeze(0),
            n_way, n_shot, 0.0
        )

        # second pass with task embedding
        encode_result = self.resnet12(x, task_embedding=emb_task, n_expand=x.size(0))
        return [res.unsqueeze(0) for res in encode_result]


class ResNet18(nn.Module):
    def __init__(self, emb_size, block=ResNet18Bottleneck):
        super(ResNet18, self).__init__()
        cfg = [64, 128, 256, 512]
        layers = [2, 2, 2, 2]
        self.emb_size = emb_size
        self.inplanes = iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], cfg[0], layers[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1], layers[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2], layers[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3], layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2)
        self.layer_second = nn.Sequential(nn.Linear(in_features=cfg[2]*5*5,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, block_num):
        layers = []
        layers.append(block(inplanes, planes, True))
        for i in range(1, block_num):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        inter = self.layer3(x)
        x = self.layer4(inter)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer_last(x)
        out = []
        out.append(x)
        inter = self.maxpool(inter)
        inter = inter.view(inter.size(0), -1)
        inter = self.layer_second(inter)
        out.append(inter)
        # no FC here
        return out


class WRN(nn.Module):

    def __init__(self, emb_size, block=WRNBottleneck, layers=[4, 4, 4]):
        super(WRN, self).__init__()
        cfg = [120, 240, 480]
        self.emb_size = emb_size
        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(3)
        self.layer_second = nn.Sequential(nn.Linear(in_features=cfg[1]*block.expansion*9,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[2]*block.expansion,
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        inter = self.layer2(x)
        x = self.layer3(inter)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer_last(x)
        inter = self.maxpool(inter)
        inter = inter.view(inter.size(0), -1)
        inter = self.layer_second(inter)
        out = []
        out.append(x)
        out.append(inter)
        # no FC here
        return out


class ConvNet(nn.Module):
    """
    Conv4 backbone
    """
    def __init__(self, emb_size, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 25 if not cifar_flag else self.hidden
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.max = nn.MaxPool2d(kernel_size=2)
        self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2,
                                          out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        output_data0 = self.max(out_3)
        out = []
        out.append(self.layer_last(output_data.view(output_data.size(0), -1)))
        out.append(self.layer_second(output_data0.view(output_data0.size(0), -1)))
        return out
