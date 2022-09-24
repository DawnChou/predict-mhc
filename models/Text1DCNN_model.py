import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F


class TextCNN1d(nn.Module):
    """
    TextCNN(1d)
    class TextCNN(nn.Module):
    """

    def __init__(self, n_filters, filter_sizes, embedding_dim, n_classes, dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 卷积核list
        """
        1d的，
        in_channels=self.embedding_dim，词向量维度个的特征
        out_channels，输出是n_filters（因为有n个卷积核在做卷积，有n个feature-map）
        kernel_size，卷积核大小（只有横向的定义的窗口大小，没有纵向）        
        """
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded):

        # embedded == [batch_size, seq_len, emb_dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded == [batch_size, emb_dim, seq_len]

        # 卷积（多个卷积核）
        """
        embedded，做每个conv，并激活
        
        变化：
        [batch_size,    emb_dim,        seq_len                         ]
        [batch_size,    n_filters,      (seq_len - filter_sizes[n] + 1) ]
        """
        conved = [
            F.relu(conv(embedded))
            for conv in self.convs
        ]

        # conved_n == [batch_size, n_filters, (seq_len - filter_sizes[n] + 1)]

        # 池化（最大池化）
        """
        conv，
        将conv.shape[2]（也就是卷积核窗口大小）做max_pool1d，
        压扁为1（现在n_filters维的数就是池化后的值），
        再去掉
        """
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conved
        ]

        # pooled_n == [batch_size, n_filters]

        # 拼接所有池化层，然后dropout
        """
        第2维变成了一张（n个卷积核*一个卷积核大小）的平面
        """
        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat == [batch_size, n_filters * len(filter_sizes)]

        # 线性映射，len(filter_sizes) * n_filters到output_dim
        out = self.fc(cat)

        # out == [batch_size, output_dim]

        return out

class DeepLION(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size, ins_num, drop_out):
        super(DeepLION, self).__init__()
        self.aa_num = aa_num
        self.feature_num = feature_num
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.ins_num = ins_num
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(self.filter_num), 1)
        self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.fc(out))
        out = out.reshape(-1, self.ins_num)
        out = self.dropout(self.fc_1(out))
        return out
