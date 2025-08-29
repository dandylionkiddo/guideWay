import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..nn.ops import Mlp

class SegformerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, num_classes, embedding_dim=256, dropout_ratio=0.1):
        super(SegformerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # self.linear_c4 = Mlp(in_features=c4_in_channels, hidden_features=embedding_dim, out_features=embedding_dim)
        # self.linear_c3 = Mlp(in_features=c3_in_channels, hidden_features=embedding_dim, out_features=embedding_dim)
        # self.linear_c2 = Mlp(in_features=c2_in_channels, hidden_features=embedding_dim, out_features=embedding_dim)
        # self.linear_c1 = Mlp(in_features=c1_in_channels, hidden_features=embedding_dim, out_features=embedding_dim)
        # .proj 서브모듈을 가지도록 구조 변경
        self.linear_c4 = nn.ModuleDict({'proj': nn.Linear(c4_in_channels, embedding_dim)})
        self.linear_c3 = nn.ModuleDict({'proj': nn.Linear(c3_in_channels, embedding_dim)})
        self.linear_c2 = nn.ModuleDict({'proj': nn.Linear(c2_in_channels, embedding_dim)})
        self.linear_c1 = nn.ModuleDict({'proj': nn.Linear(c1_in_channels, embedding_dim)})

        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(embedding_dim),
        #     nn.ReLU(inplace=True)
        # )
        # 체크포인트와 맞추기 위해 named 레이어로 변경
        self.linear_fuse = nn.ModuleDict({
            'conv': nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=1, bias=False),
            'bn': nn.BatchNorm2d(embedding_dim)
        })

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # conv_seg, dropout도 추가 (체크포인트에 있으니까)
        # linear_pred: 입력 채널로 embedding_dim (256) 사용, conv_seg: 입력 채널로 128을 직접 지정
        self.conv_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # MLP decoder on C1-C4 feature map
        n = c4.shape[0]
        # # c4: (B, C4, H4, W4) -> (B, H4*W4, C4) -> MLP -> (B, H4*W4, embed_dim) -> (B, embed_dim, H4, W4)
        # B, _, H4, W4 = c4.shape
        # _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2), H4, W4)\
        #             .permute(0, 2, 1).reshape(B, -1, H4, W4)
        # _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        # B, _, H3, W3 = c3.shape
        # _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2), H3, W3)\
        #             .permute(0, 2, 1).reshape(B, -1, H3, W3)
        # _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        # B, _, H2, W2 = c2.shape
        # _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2), H2, W2)\
        #             .permute(0, 2, 1).reshape(B, -1, H2, W2)
        # _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        # B, _, H1, W1 = c1.shape
        # _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2), H1, W1)\
        #             .permute(0, 2, 1).reshape(B, -1, H1, W1)

        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # c4 처리 - .proj 서브모듈 사용
        B, _, H4, W4 = c4.shape
        _c4 = c4.permute(0, 2, 3, 1).contiguous()  # (B, H4, W4, C4)
        _c4 = self.linear_c4['proj'](_c4)  # proj
        _c4 = _c4.permute(0, 3, 1, 2).contiguous()  # (B, embedding_dim, H4, W4)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # c3 처리 - .proj 서브모듈 사용
        B, _, H3, W3 = c3.shape
        _c3 = c3.permute(0, 2, 3, 1).contiguous()
        _c3 = self.linear_c3['proj'](_c3)  # proj
        _c3 = _c3.permute(0, 3, 1, 2).contiguous()
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # c2 처리 - .proj 서브모듈 사용
        B, _, H2, W2 = c2.shape
        _c2 = c2.permute(0, 2, 3, 1).contiguous()
        _c2 = self.linear_c2['proj'](_c2)  # proj
        _c2 = _c2.permute(0, 3, 1, 2).contiguous()
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # c1 처리 - .proj 서브모듈 사용
        B, _, H1, W1 = c1.shape
        _c1 = c1.permute(0, 2, 3, 1).contiguous()
        _c1 = self.linear_c1['proj'](_c1)  # proj
        _c1 = _c1.permute(0, 3, 1, 2).contiguous()

        # Feature fusion - named 레이어 사용
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse['conv'](_c)  # conv
        _c = self.linear_fuse['bn'](_c)    # bn  
        _c = F.relu(_c, inplace=True)      # relu

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # x = self.conv_seg(x)  # 모델 가중치를 성공적으로 불러오기 위해 존재하는 구조적 장치

        return x
