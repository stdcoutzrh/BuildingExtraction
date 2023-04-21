import torch
import torch.nn as nn
from functools import partial

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import copy

tiny = [1,2,4,1]

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.dim = dim
        if self.dim_scale!=1:
            self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
            self.conv = nn.Conv2d(dim//2, dim//2, 3,1,1,1,16) if dim_scale==2 else nn.Identity()
            self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H = int(math.sqrt(x.shape[1]))
        W =  H 
        
        if self.dim_scale == 1:
            return x,H,W
        else:
            x = self.expand(x)
            B, L, C = x.shape

            x = x.view(B, H, W, C).contiguous() 
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)# b h w c

            x = self.conv(x.permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous().view(B,-1,C//4).contiguous() 
            x= self.norm(x)
            return x,H*2,W*2

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm,num_classes=6):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

        self.out = nn.Sequential(
                nn.Conv2d(in_channels=self.dim,out_channels=self.dim,kernel_size=3,padding=1,bias=False,groups=16),
                nn.Conv2d(in_channels=self.dim,out_channels=num_classes,kernel_size=1,bias=False),
            ) 

    def forward(self, x):
        H = int(math.sqrt(x.shape[1]))
        W = H
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, H, W, C).contiguous() 
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)

        x = x.permute(0,3,1,2).contiguous()
        x = self.out(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1,
                return_attnmap=False):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.return_attnmap = return_attnmap

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() 
        if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  #torch.Size([1, 64, 128, 128])
                
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()  #torch.Size([1, 256, 64])
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()  #torch.Size([1, 1024, 64])
                
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
                k1, v1 = kv1[0], kv1[1] 
                k2, v2 = kv2[0], kv2[1] 
                
                attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)   
                v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2).contiguous()  
                
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

                x = torch.cat([x1,x2], dim=-1)

                if self.return_attnmap:
                    attn = torch.cat([attn1,attn2],dim=3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2).contiguous() 
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.return_attnmap:
            return x,attn.mean(1)
        else:
            return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 return_attnmap=False):
        super().__init__()
        
        self.return_attnmap = return_attnmap

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,return_attnmap=self.return_attnmap)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W): 
        if self.return_attnmap:     
            x_,attnmap = self.attn(self.norm1(x), H, W)
        else:
            x_=self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        if self.return_attnmap:
            return x,attnmap
        else:
            return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, 64, 7, 2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)]
        for i in range(num):
            stem.append(nn.Conv2d(64, 64, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(64))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(64, 64, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(64)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H,W

class ShuntedTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], # t [1,2,4,1]
                 sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better
    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous() 
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class ShuntedTransformerSys(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, num_classes=6, 
                 embed_dims=[64, 128, 256, 512],
                 embed_dims_up=[512, 256, 128, 64],
                 num_heads=[2, 4, 8, 16],
                 num_heads_up = [16,8,4,2],
                 mlp_ratios=[8, 8, 4, 4], 
                 mlp_ratios_up=[4,4,8, 8], 
                 
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,

                 depths=[1,2,4,1], 
                 depths_up = [1,4,2,1],
                 sr_ratios=[8, 4, 2, 1],
                 sr_ratios_up=[1, 2, 4, 8],
                 
                 num_stages=4, 
                 num_conv=0):
        super().__init__()
        self.num_classes = num_classes

        # down--------------------------------------------------------
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i],return_attnmap=True)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # up--------------------------------------------------------
        self.up_depths = depths_up
        self.up_num_stages = num_stages

        dpr_up = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.up_depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.up_num_stages):
            if i ==0:
                patch_expand = PatchExpand(embed_dims_up[i],dim_scale=1)
            else:
                patch_expand = PatchExpand(embed_dims_up[i-1],dim_scale=2)

            block = nn.ModuleList([Block(
                dim=embed_dims_up[i], 
                num_heads=num_heads_up[i], 
                mlp_ratio=mlp_ratios_up[i], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr_up[len(dpr_up)-1 - cur  - j], 
                norm_layer=norm_layer,
                sr_ratio=sr_ratios_up[i],return_attnmap=False)  for j in range(depths_up[i])])   # [1,2,4,1]
            norm = norm_layer(embed_dims_up[i])
            cur += depths_up[i]
            
            if i!=0:
                attn_conv = nn.Conv2d(1280,embed_dims_up[i],1,1,0,1,64)
                rate1 = torch.nn.Parameter(torch.Tensor(1))
                rate2 = torch.nn.Parameter(torch.Tensor(1))
                rate3 = torch.nn.Parameter(torch.Tensor(1))
                rate1.data.fill_(1.0)
                rate2.data.fill_(1.0)
                rate3.data.fill_(0.1)
                setattr(self,f"up_rate1_{i+1}",rate1)
                setattr(self,f"up_rate2_{i+1}",rate2)
                setattr(self,f"up_rate3_{i+1}",rate3)
                setattr(self,f"up_attnconv_{i+1}",attn_conv)
                #print("up_attnconv_{i+1}")
            
            setattr(self, f"patch_expand{i + 1}", patch_expand)
            setattr(self, f"up_block{i + 1}", block)
            setattr(self, f"up_norm{i + 1}", norm)

        self.out = FinalPatchExpand_X4(dim=64,num_classes=self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        fms = []
        attns = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
           
            x, H, W = patch_embed(x)

            for blk in block:
                x,attn= blk(x, H, W)
            x = norm(x)    
            if i != self.num_stages - 1:
                fms.append(x)   # B L C
                attns.append(attn)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x,fms,attns

    def forward_up_features(self, x,fms,attn_maps):
        B = x.shape[0]
        for i in range(self.up_num_stages):

            patch_expand = getattr(self, f"patch_expand{i + 1}")
            block = getattr(self, f"up_block{i + 1}")
            norm = getattr(self, f"up_norm{i + 1}")

            if i!=0:
                rate1 = getattr(self,f"up_rate1_{i+1}")
                rate2 = getattr(self,f"up_rate2_{i+1}")
                rate3 = getattr(self,f"up_rate3_{i+1}")
                attn_conv = getattr(self,f"up_attnconv_{i + 1}")
                attns = attn_conv(attn_maps[self.num_stages - i -1].permute(0,2,1).contiguous().unsqueeze(2)).squeeze(2).permute(0,2,1).contiguous()
                attns = torch.softmax(attns,-2)
            x, H, W = patch_expand(x)

            if i != 0:
                x = x* rate1 + fms[self.num_stages - i -1]*rate2 + attns#*rate3#add1
            for blk in block:
                x = blk(x, H, W)

            x = norm(x)            
        return x

    def forward(self, x):
        x,fms,attnmaps= self.forward_features(x)    
        x_up = self.forward_up_features(x,fms,attnmaps)     
        x_out = self.out(x_up)
        return x_out

class ShuntedUnet(nn.Module):
    def __init__(self,img_size=512, num_classes=6, zero_head=False, vis=False):
        super(ShuntedUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.shunted_unet = ShuntedTransformerSys(patch_size=4,num_classes=self.num_classes, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16],num_heads_up=[16, 8, 4, 2], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
         depths = tiny, 
         sr_ratios=[8, 4, 2, 1], 
         num_conv=0)
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.shunted_unet(x)
        return logits

    def load_from(self, pretrained_path='/home/zrh/code/data/ckpt_T.pth'):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            #new_pretrained_dict = {"shunted_unet."+k:v for k,v in pretrained_dict.items()}
            
            model_dict = self.shunted_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "block" in k:
                    current_layer_num = 5-int(k[5])
                    current_k = "up_block" + str(current_layer_num) + k[9:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            print(len(full_dict))
            print(len(model_dict))
            self.shunted_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")

def create_shunted_unet_model(pretrained=True,
        ckpt_path='/home/zrh/code/ckpt/ckpt_T.pth',n_classes=6):
    model = ShuntedUnet(512,n_classes)

    total_paramters = 0
    for  parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
    print("current model Params: %.4f M" % (total_paramters / 1e6))

    if pretrained:
        model.load_from(ckpt_path)
    return model

if __name__ == "__main__":


    model = create_shunted_unet_model(pretrained=False,
        ckpt_path='/mnt/sda/zrh/code/pretrained_ckpt/ckpt_T.pth',n_classes=6)
    x = torch.ones([1,3,512,512])

    res = model(x)
    print(res.shape)

    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, x)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 

        # FLOPs: 29.8119 G
        # Params: 21.3197 M
    