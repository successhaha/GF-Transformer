import mix_transformer
import torch.nn as nn
from torch.nn import Conv2d, Parameter, Softmax
import torch
import torch.nn.functional as F
import numpy as np
class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)
class ConvRelu_normal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu_normal, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride = 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)
class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(mix_transformer, backbone)()
        ## initilize encoder
        if pretrained:
            state_dict = torch.load(backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )
def Encoder():
    model = Transformer("mit_b3", pretrained=True)
    return model.cuda()
class Decoder(nn.Module):
    def __init__(self, num_classes = 1, decoder_dim = 256):
        super(Decoder, self).__init__()
        encoder_filters = [64, 128, 320, 512]
        decoder_filters = np.asarray([32, 64, 128, 256])

        self.conv1 = ConvRelu(encoder_filters[-1], decoder_filters[-1]) # 512-->256
        self.conv1_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1]) # 256+320-->256
        self.conv2 = ConvRelu(decoder_filters[-1], decoder_filters[-2]) # 256-->160
        self.conv2_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2]) # 160+128-->128
        self.conv3 = ConvRelu(decoder_filters[-2], decoder_filters[-3])  # 160->64
        self.conv3_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])  # 64+64-->64
        self.conv4 = ConvRelu(decoder_filters[-3], decoder_filters[-4])  # 64-->32

        self.cat1 = ConvRelu_normal(encoder_filters[-4], encoder_filters[-3])
        self.cat2 = ConvRelu_normal(encoder_filters[-3], encoder_filters[-2])
        self.cat3 = ConvRelu_normal(encoder_filters[-2], encoder_filters[-1])
        self.cat4 = ConvRelu(4*encoder_filters[-1], encoder_filters[-1])
        self.res = nn.Conv2d(decoder_filters[-4], 1, 1, stride=1, padding=0)
    def forward(self, out):
        con_cat1 = self.cat3(self.cat2(self.cat1(out[0])))
        con_cat2 = self.cat3(self.cat2(out[1]))
        con_cat3 = self.cat3(out[2])
        con_cat4 = self.cat4(torch.cat([con_cat1, con_cat2, con_cat3, out[3]], 1))
        dec1 = self.conv1(F.interpolate(torch.add(out[3], con_cat4), scale_factor=2))
        dec1 = self.conv1_2(torch.cat([dec1, out[2]], 1))


        dec2 = self.conv2(F.interpolate(dec1,scale_factor=2))
        dec2 = self.conv2_2(torch.cat([dec2, out[1]], 1))


        dec3 = self.conv3(F.interpolate(dec2, scale_factor=2))
        dec3 = self.conv3_2(torch.cat([dec3, out[0]], 1))


        dec4 = self.conv4(F.interpolate(dec3, scale_factor=4))
        return self.res(dec4)
class GFformer_one(nn.Module):
    def __init__(self):
        super(GFformer_one, self).__init__()
        model = Encoder()

        self.encoder = model.encoder
        self.decoder = Decoder()

    def forward(self, pre_img):
        B = pre_img.shape[0]
        #stage1
        r1, H1, W1 = self.encoder.patch_embed1(pre_img)
        for i, blk in enumerate(self.encoder.block1):
            r1 = blk(r1, H1, W1)
        r1 = self.encoder.norm1(r1)
        r1 = r1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        #stage2
        r2, H2, W2 = self.encoder.patch_embed2(r1)
        for i, blk in enumerate(self.encoder.block2):
            r2 = blk(r2, H2, W2)
        r2 = self.encoder.norm2(r2)
        r2 = r2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        # stage3
        r3, H3, W3 = self.encoder.patch_embed3(r2)
        for i, blk in enumerate(self.encoder.block3):
            r3 = blk(r3, H3, W3)
        r3 = self.encoder.norm3(r3)
        r3 = r3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        # stage4
        r4, H4, W4 = self.encoder.patch_embed4(r3)
        for i, blk in enumerate(self.encoder.block4):
            r4 = blk(r4, H4, W4)
        r4 = self.encoder.norm4(r4)
        r4 = r4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        out = [r1, r2, r3, r4]
        out = self.decoder(out)
        return out
   class Decoder_double(nn.Module):
    def __init__(self, num_classes = 1, decoder_dim = 256):
        super(Decoder_double, self).__init__()
        encoder_filters = [64, 128, 320, 512]
        decoder_filters = np.asarray([32, 64, 128, 256])
        
        self.conv1 = ConvRelu(encoder_filters[-1], decoder_filters[-1])                         # 512-->256
        self.conv1_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1]) # 256+320-->256
        self.conv2 = ConvRelu(decoder_filters[-1], decoder_filters[-2])                         # 256-->128
        self.conv2_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2]) # 128+128-->128
        self.conv3 = ConvRelu(decoder_filters[-2], decoder_filters[-3])  # 128->64
        self.conv3_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])  # 64+64-->64
        self.conv4 = ConvRelu(decoder_filters[-3], decoder_filters[-4])  # 64-->32

        self.res = nn.Conv2d(decoder_filters[-4], 5, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward1(self, out):
        dec1 = self.conv1(F.interpolate(out[3], scale_factor = 2))
        dec1 = self.conv1_2(torch.cat([dec1, out[2]], 1))


        dec2 = self.conv2(F.interpolate(dec1, scale_factor=2))
        dec2 = self.conv2_2(torch.cat([dec2, out[1]], 1))

        dec3 = self.conv3(F.interpolate(dec2, scale_factor=2))
        dec3 = self.conv3_2(torch.cat([dec3, out[0]], 1))

        dec4 = self.conv4(F.interpolate(dec3, scale_factor=4))
        dec4 = self.relu(dec4)
        return dec4
    def forward(self, out):
        dec5 = self.forward1(out)
        return self.res(dec5)
class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)
   class GFM(nn.Module):
    def __init__(self, in_c):
        super(GFM, self).__init__()
        self.conv_n1 = convblock(in_c, in_c//2, 1, 1, 0)
        self.conv_n2 = convblock(in_c, in_c//2, 1, 1, 0)
        self.conv_n3 = convblock(in_c, in_c//2, 1, 1, 0)
        self.conv_rt = convblock(in_c//2, in_c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.gam1 = nn.Parameter(torch.zeros(1))
        self.gam2 = nn.Parameter(torch.zeros(1))

    def forward(self, pre, post, glo):
        b, c, h, w = pre.size()
        pre1 = self.conv_n1(pre)
        pre2 = pre1.view(b, -1, w * h)
        glo1 = self.conv_n2(glo).view(b, -1, w * h)
        post2 = self.conv_n3(post).view(b, -1, w * h)

        post_2 = post2.permute(0, 2, 1)
        pre_1 = pre2.permute(0, 2, 1)
        pre_glo = torch.bmm(pre_1, glo1)
        post_glo = torch.bmm(post_2, glo1)
        rt_glo = pre_glo + post_glo
        softmax_rt = self.softmax(rt_glo)

        att_post = torch.bmm(post2, softmax_rt)
        att_pre = torch.bmm(pre2, softmax_rt)
        b1, c1, h1, w1 = pre1.size()

        rt_post = att_post.view(b1, c1, h1, w1)
        rt_pre = att_pre.view(b1, c1, h1, w1)

        out_post = self.conv_rt(rt_post)
        out_pre = self.conv_rt(rt_pre)
        out_pre = self.gam1 * out_pre + pre
        out_post = self.gam2 * out_post + post
        return out_pre, out_post
    
