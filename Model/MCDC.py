import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from Model.wavemlp import WaveMLP_M
import math
from thop import profile
from Model.res2net50_26w_4s import res2net50_26w_4s as resnet50


groups = 32

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features).cuda()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features).cuda()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = up_conv(in_channels, out_channels)
        #self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            # coorAtt(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 = self.att_block(x1, x2)
        x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down,left2):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)
        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')
        z2 = F.relu(down_mask * left, inplace=True)
        out = torch.cat((z1, z2,left2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class Cross_ATT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = self.dim  // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(self.dim ,  self.dim * 2, bias=qkv_bias)
        self.attnq_drop = nn.Dropout(0.2)
        self.attnk_drop = nn.Dropout(0.2)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q = nn.Linear( self.dim ,  self.dim , bias=qkv_bias)
        self.A = nn.Linear(self.dim, self.dim, bias=qkv_bias)

    def forward(self,x,x2):
        B, N, C = x.shape
        q=self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        A=self.A(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        ATT_q = (q @ A.transpose(-2, -1)) * self.scale #Me
        ATT_q=ATT_q.softmax(dim=-1)
        ATT_q = self.attnq_drop(ATT_q)
        ATT_k = (A @ k.transpose(-2, -1)) * self.scale #Md
        ATT_k=ATT_k.softmax(dim=-1)
        ATT_k = self.attnk_drop(ATT_k)
        Out=ATT_q@(ATT_k@v)
        Out=Out.transpose(1,2).reshape(B,N,C)
        Out = self.proj_drop(Out)
        return Out

class CSGF(nn.Module):
    #cross-scale global feature fusion
    def __init__(self, in_dim=64,out_dim=128,num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, attn_drop=0.2,
                 drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(out_dim)
        self.attn = Cross_ATT(out_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 =norm_layer(out_dim)
        self.norm3=norm_layer(out_dim)
        mlp_hidden_dim = int((out_dim) * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.up = up_conv(in_dim, out_dim)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_dim*2, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self,x1,x2):
        B, C, H, W = x2.shape
        x2 = rearrange(x2, 'b c h w  -> b (h w) c ')  # [B,N,C]
        x1 = self.up(x1)  # [B,C,H,W]
        x1 = rearrange(x1, 'b c h w  -> b (h w) c ')  # [B,N,C]
        x1 = x1 + self.drop_path(self.attn(self.norm1(x1),self.norm3(x2)))
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=H, w=W)  # [B,C,H,W]
        return x1

class SMFF(nn.Module):
    #Segmentation mask feature fusion module
    def __init__(self, C_channels, T_channels, r=4):
        super(SMFF, self).__init__()
        inter_channels = int(C_channels // r)

        # 特征对齐
        self.local_att = nn.Sequential(
            nn.Conv2d(T_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, T_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d( T_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(T_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels,  T_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d( T_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        xa = x1 + x2
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x1 * wei + 2 * x2 * (1 - wei)
        return xo


class ConvModule(nn.Module):
    def __init__(self, In_channels, Out_channels,kernel_size,stride,padding):
        super(ConvModule, self).__init__()
        self.convmodule = nn.Sequential(
            nn.Conv2d(In_channels, Out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d( Out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x=self.convmodule(x)
        return x

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels,r=2):
        super(ChannelAtt, self).__init__()
        inter_channels = int(in_channels // r)
        self.global_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_1x1 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward function."""
        feat = self.global_att(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten

class DPCM(nn.Module):
    def __init__(self, C_channels,T_channels,r=16):
        super(DPCM, self).__init__()
        # Dual-path complementary module
        self.match = nn.Sequential(
            nn.Conv2d(T_channels, C_channels,kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(C_channels)
        )
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.spatial_mlp = nn.Sequential(nn.Linear(256, C_channels), nn.ReLU(), nn.Linear(C_channels, C_channels))
        self.spatial_att = ChannelAtt(C_channels , C_channels)
        self.context_mlp = nn.Sequential(nn.Linear(256, C_channels), nn.ReLU(), nn.Linear(C_channels, C_channels))
        self.context_att = ChannelAtt(C_channels, C_channels)
        self.smooth = ConvModule(C_channels, C_channels, 3, stride=1, padding=1)

    def forward(self, sp_feat, co_feat):
        co_feat = self.match(co_feat)
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b, c, h, w = s_att.size()  #b,c,1,1
        s_att_split = s_att.view(b, self.r, c // self.r)  #b,r,c//r
        c_att_split = c_att.view(b, self.r, c // self.r)  #b,r,c//r
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))#b,r,r
        chl_affinity = chl_affinity.contiguous().view(b, -1)#b,256
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity)) #b,c
        co_mlp_out = F.relu(self.context_mlp(chl_affinity)) #b,c
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1)) #b,c,1,1
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att) #b,c,h,w
        out = self.smooth(s_feat + c_feat)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if self.attn_block is not None:
            x2 = self.attn_block(x1, x2)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MCDC(nn.Module):
    def __init__(self, dim, n_class, in_ch=3):
        super().__init__()

        self.encoder = resnet50()
        self.encoder.fc = nn.Identity()
        self.encoder.layer4 = nn.Identity()

        self.encoder2 =WaveMLP_M()  # [64, 128, 320, 512]
        path = '/datasets/Dset_Jerry/Pre_train/WaveMLP_M.pth.tar'
        save_model = torch.load(path)
        model_dict = self.encoder2.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.encoder2.load_state_dict(model_dict)
        self.drop = nn.Dropout2d(0.2)

        #Encoder fusion: Dual-path complementary module
        self.dpcm1 = DPCM(256, 64)
        self.dpcm2 = DPCM(512, 128)
        self.dpcm3 = DPCM(1024, 320)

        # decoder CNN
        self.decoder2_1 = CSLF(16* dim,8 * dim, nb_Conv=2)
        self.decoder2_2 = CSLF(8 * dim,4* dim, nb_Conv=2)

        #decoder2 Trans
        self.decoder_1 = CSGF(16* dim,8 * dim)
        self.decoder_2= CSGF(8 * dim,4 * dim)

        self.loss0 = nn.Sequential(
            Conv(dim, dim, 3, bn=True, relu=True),
            Conv(dim, n_class, 3, bn=False, relu=False)
        )

        self.loss1 = nn.Sequential(
            Conv(16 * dim, 4 * dim, 1, bn=True, relu=True),
            Conv(4 * dim, 4 * dim, 3, bn=True, relu=True),
            Conv(4 * dim, n_class, 3, bn=False, relu=False)
        )
        # Decoder fusion: Segmentation mask feature fusion module
        self.output = SMFF(4 * dim, 4 * dim)
        self.loss2 = nn.Sequential(
            Conv(4*dim, 4*dim, 3, bn=True, relu=True),
            Conv(4*dim, n_class, 3, bn=False, relu=False)
        )

    def forward(self, x):

        # top-down path
        #encoder
        c0 = self.encoder.conv1(x)
        c0 = self.encoder.bn1(c0)
        c0 = self.encoder.relu(c0) #[B,64,H/2,W/2]
        c1 = self.encoder.maxpool(c0)
        c1 = self.encoder.layer1(c1)
        c1 = self.drop(c1)  #[B,256,H/4,W/4]
        c2 = self.encoder.layer2(c1)
        c2 = self.drop(c2)  #[B,512,H/8,W/8]
        c3 = self.encoder.layer3(c2)
        c3 = self.drop(c3)  #[B,1024,H/16,W/16]

        #encoder2
        out2 = self.encoder2(x)
        t1, t2, t3, t4 = out2[0], out2[1], out2[2], out2[3]
        #[B,64,H/4,W/4] [B,128,H/8,W/8] [B,320,H/16,W/16] [B,512,H/32,W/32]
        loss0 = F.interpolate(self.loss0(t1), scale_factor=4, mode='bilinear', align_corners=True)

        # fusion
        ct1= self.dpcm1(c1,t1)  #[B,256,H/4,W/4]
        ct2 =self.dpcm2(c2,t2) #[B,512,H/8,W/8]
        ct3 =self.dpcm3(c3,t3)   #[B,1024,H/16,W/16]
        loss1= F.interpolate(self.loss1(ct3), scale_factor=16, mode='bilinear', align_corners=True)

        # down-top path
        # decoder CNN
        d_1=self.decoder2_1(ct3,ct2) #[B,128,H/8,W/8]
        d_2 = self.decoder2_2(d_1, ct1) #[B,64,H/4,W/4]

        #decoder2 Trans
        d2_1 = self.decoder_1(ct3,ct2) #[B,128,H/8,W/8]
        d2_2 = self.decoder_2(d2_1,ct1) #[B,64,H/4,W/4]

        #output fusion
        loss2=F.interpolate(self.loss2(self.output(d_2,d2_2)), scale_factor=4, mode='bilinear', align_corners=True)

        #CAM

        return loss0,loss1,loss2

class CSLF(nn.Module):
    #cross-scale local feature fusion
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = up_conv(in_channels, out_channels)
        self.coatt = CCA(F_g=out_channels, F_x=out_channels)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(2*out_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(1, 3, 256,256)).cuda()
    model = UNet(64, 1).cuda()


