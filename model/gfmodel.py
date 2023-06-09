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
class GF_module(nn.Module):
    def __init__(self, in_c, in_g):
        super(GF_module, self).__init__()
        self.conv_fus1 = convblock(2*in_c, in_c, 3, 1, 1)
        self.conv_fus2 = convblock(2*in_c, in_c, 3, 1, 1)
        self.conv_out = convblock(in_c, in_g, 3, 1, 1)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_c, in_c, 1, 1, 0),
            nn.Sigmoid()
        )
        self.ca = CA(64)
        self.sig = nn.Sigmoid()
        self.conv_r = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_t = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, rgb, t):
        rgb = self.ca(rgb)
        t   = self.ca(t)
        fus_t   = self.conv_fus1(torch.cat((torch.add(rgb, torch.mul(rgb, self.rt_fus(t))), t), 1))
        fus_rgb = self.conv_fus2(torch.cat((torch.add(t, torch.mul(t, self.rt_fus(rgb))), rgb), 1))

        out = self.conv_out(fus_rgb + fus_t)
        return out
class CSGF(nn.Module):
    def __init__(self, in_ch, h, in_glo):
        super(CSGF, self).__init__()
        self.down = convblock(in_glo, in_ch, 3, 1, 1)
        self.gc_shuffle = Gconv_Shuffle(in_ch, 32)
        self.gc2 = Gconv(32, 2)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2*h*h, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )
    def forward(self, pre, post, glo):
        b,c,h,w = pre.size()

        cmf = self.gc_shuffle(self.down(glo))
        s_w = self.gc2(cmf)
        s_w_pre = s_w[:, 0, :, :].unsqueeze(1)
        s_w_post = s_w[:, 1, :, :].unsqueeze(1)

        c_cmf = torch.flatten(s_w, start_dim=1)
        c_cmf = self.fc(c_cmf)  # 
        c_w_pre = c_cmf[:, 0].unsqueeze(1).view(b, 1, 1, 1)
        c_w_post = c_cmf[:, 1].unsqueeze(1).view(b, 1, 1, 1)
        post = post + torch.mul(c_w_post, (torch.mul(s_w_post, post)))
        pre = pre + torch.mul(c_w_pre, torch.mul(s_w_pre, pre))

        return pre, post
class GFformer_two(nn.Module):
    def __init__(self):
        super(GFformer_two, self).__init__()
        model = Encoder()
        self.gfm1 = GF_module(64, 128)    
        self.gfm2 = GF_module1(128, 320, 33)
        self.gfm3 = GF_module1(320, 512, 22)
        self.gfm4 = GF_module1(512, 512, 11)

        self.CSGF1 = CSGF(64, 128, 128)
        self.CSGF2 = CSGF(128, 64, 320)
        self.CSGF3 = CSGF(320, 32, 512)
        # self.fusion1 = FAModule(64, 64)
        self.rgb_net = model.encoder
        self.post_net = model.encoder
        self.decoder = Decoder_double()
        decoder_filters = np.asarray([20, 128, 256, 512]) // 2
        self.res = nn.Conv2d(decoder_filters[-4], 5, 1, stride=1, padding=0)
    def forward(self, rgb):
        pre_image = rgb[:, :3, :, :]
        post_image = rgb[:, 3:, :, :]
        B = rgb.shape[0]
        #stage1
        """
        /*
        输入：灾前图像
        输出：64*128*128
        */
        """
        r1, H1, W1 = self.rgb_net.patch_embed1(pre_image)
        for i, blk in enumerate(self.rgb_net.block1):
            r1 = blk(r1, H1, W1)
        r1 = self.rgb_net.norm1(r1)
        r1 = r1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        """
        /*
        输入：灾后图像
        输出：64*128*128
        */
        """
        r1_1, H1, W1 = self.rgb_net.patch_embed1(post_image)
        for i, blk in enumerate(self.rgb_net.block1):
            r1_1 = blk(r1_1, H1, W1)
        r1_1 = self.rgb_net.norm1(r1_1)
        r1_1 = r1_1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        """
        /*
        全局引导模块
        输入：灾前和灾后图像
        输出：64*128*128
        */
        """
        global_1 = self.gfm1(r1, r1_1)

        # print("stage1 shape", r1.shape, r1_1.shape, global_1.shape)
        r1, r1_1 = self.CSGF1(r1, r1_1, global_1)
        # r1_fusion = self.gfm1(r1, global_1)
        # r1_1_fusion = self.gfm1(r1_1, global_1)

        #stage2
        r2, H2, W2 = self.rgb_net.patch_embed2(r1)
        for i, blk in enumerate(self.rgb_net.block2):
            r2 = blk(r2, H2, W2)
        r2 = self.rgb_net.norm2(r2)
        r2 = r2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        r2_1, H2, W2 = self.rgb_net.patch_embed2(r1_1)
        for i, blk in enumerate(self.rgb_net.block2):
            r2_1 = blk(r2_1, H2, W2)
        r2_1 = self.rgb_net.norm2(r2_1)
        r2_1 = r2_1.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        global_2 = self.gfm2(r2, r2_1, global_1)
        r2_fusion, r2_1_fusion = self.CSGF2(r2, r2_1, global_2)

        # stage3
        r3, H3, W3 = self.rgb_net.patch_embed3(r2_fusion)
        for i, blk in enumerate(self.rgb_net.block3):
            r3 = blk(r3, H3, W3)
        r3 = self.rgb_net.norm3(r3)
        r3 = r3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        r3_1, H3, W3 = self.rgb_net.patch_embed3(r2_1_fusion)
        for i, blk in enumerate(self.rgb_net.block3):
            r3_1 = blk(r3_1, H3, W3)
        r3_1 = self.rgb_net.norm3(r3_1)
        r3_1 = r3_1.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        global_3 = self.gfm3(r3, r3_1, global_2)
        r3_fusion, r3_1_fusion = self.CSGF3(r3, r3_1, global_3)

        # stage4
        r4, H4, W4 = self.rgb_net.patch_embed4(r3_fusion)
        for i, blk in enumerate(self.rgb_net.block4):
            r4 = blk(r4, H4, W4)
        r4 = self.rgb_net.norm4(r4)
        r4 = r4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        r4_1, H4, W4 = self.rgb_net.patch_embed4(r3_1_fusion)
        for i, blk in enumerate(self.rgb_net.block4):
            r4_1 = blk(r4_1, H4, W4)
        r4_1 = self.rgb_net.norm4(r4_1)
        r4_1 = r4_1.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()


        out = [r1, r2, r3, r4]
        out1 = [r1_1, r2_1, r3_1, r4_1]
        out = self.decoder(out)
        out_1 = self.decoder(out1)

        dec5 = torch.cat([out, out_1], 1)
        return self.res(dec5)
