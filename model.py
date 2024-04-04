import math
import torch.nn as nn
import torch
from images.image_models import ImageEncoder
from text.text_models import TextEncoder, TextEncoder_without_know
from interraction.inter_models import CroModality
import utils.gat as tg_conv
import torch.nn.functional as F


class TextDiff(nn.Module):
    def __init__(self,input_size=300, dropout=0):
        super(TextDiff, self).__init__()
        self.multiheadattention = nn.MultiheadAttention(embed_dim=input_size, num_heads=5, dropout=dropout)
        self.multiheadattention1 = nn.MultiheadAttention(embed_dim=input_size, num_heads=5, dropout=dropout)
        self.laynorm1 = nn.LayerNorm(input_size)
        self.laynorm2 = nn.LayerNorm(input_size)

    def forward(self, text, caption, mask):
        row_text = torch.cat([text,caption], dim=1)
        text = self.multiheadattention(row_text, row_text, row_text, key_padding_mask=mask)[0]
        text = self.laynorm1( text + row_text )
        text1 = text
        text = self.multiheadattention(text, text, text, key_padding_mask=None)[0]
        text = self.laynorm2(text+text1)
        return text

class Alignment(nn.Module):
    def __init__(self, input_size=300, txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False,
                 img_gat_layer=2, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, is_knowledge=0):
        super(Alignment, self).__init__()

        self.input_size = input_size
        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops

        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.is_knowledge = is_knowledge

        self.txt_conv = nn.ModuleList(
            [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
                             concat=False, dropout=self.txt_gat_drop, fill_value="mean",
                             add_self_loops=self.txt_self_loops, is_text=True)
             for i in range(self.txt_gat_layer)])

        if self.is_knowledge == 0:
            self.img_conv = nn.ModuleList([tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size,
                                                           heads=self.img_gat_head, concat=False,
                                                           dropout=self.img_gat_drop, fill_value="mean",
                                                           add_self_loops=self.img_self_loops) for i in
                                           range(self.img_gat_layer)])
        else:
            self.img_conv = nn.ModuleList([tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size,
                                                           heads=self.img_gat_head, concat=False,
                                                           dropout=self.img_gat_drop, fill_value="mean",
                                                           add_self_loops=self.img_self_loops, is_text=True) for i in
                                           range(self.img_gat_layer)])

        # for token compute the importance of each token
        self.linear1 = nn.Linear(self.input_size, 1)
        # for np compute the importance of each np
        self.linear2 = nn.Linear(self.input_size, 1)
        self.norm = nn.LayerNorm(self.input_size)
        self.relu1 = nn.ReLU()

    def forward(self, t2, v2, edge_index, gnn_mask, score, key_padding_mask, np_mask, img_edge_index,
                gnn_mask_know=None, lam=1):
        """

        Args:
            v2: (N,K,D)
            t2: (N,L,D)
            edge_index: (N,2)
            gnn_mask:(N). Tensor on gpu. If ture, the graph is masked.
            score: (N,L,D). The importance of each word or np. Computed by text encoder
            key_padding_mask: (N,L) Tensor. L is the np and word length. True means mask
            np_mask: (N,L+1) Tensor. L is the np and word length. True means mask, if np, Flase.

        Returns:
            a:(N, K) alignment distribution
        """
        # congruity score of atomic level
        q1 = torch.bmm(t2, v2.permute(0, 2, 1)) / math.sqrt(t2.size(2))
        c = torch.sum(score * t2, dim=1, keepdim=True)
        # (N,token_length)
        pa_token = self.linear1(t2).squeeze().masked_fill_(key_padding_mask, float("-Inf"))

        tnp = t2
        # for node without edge, it representation will be zero-vector
        for gat in self.txt_conv:
            tnp = self.norm(torch.stack(
                [(self.relu1(gat(data[0], data[1].cuda(), mask=data[2]))) for data in zip(tnp, edge_index, gnn_mask)]))
        # 尝试给knowledge加上
        v3 = v2
        if self.is_knowledge == 0:
            for gat in self.img_conv:
                v3 = self.norm(torch.stack([self.relu1(gat(data, img_edge_index.cuda())) for data in v3]))
        else:
            for gat in self.img_conv:
                v3 = torch.stack([self.relu1(gat(data[0].cuda(), data[1].cuda(), mask=data[2]))
                                  for data in zip(v3, img_edge_index, gnn_mask_know)])
        tnp = torch.cat([tnp, c], dim=1)
        #  congruity score of compositional level
        q2 = torch.bmm(tnp, v3.permute(0, 2, 1)) / math.sqrt(tnp.size(2))

        pa_np = self.linear2(tnp).squeeze().masked_fill_(np_mask, float("-Inf"))
        pa_np = nn.Softmax(dim=1)(pa_np * lam).unsqueeze(2).repeat((1, 1, v3.size(1)))
        pa_token = nn.Softmax(dim=1)(pa_token * lam).unsqueeze(2).repeat((1, 1, v3.size(1)))
        a_1 = torch.sum(q1 * pa_token, dim=1)
        a_2 = torch.sum(q2 * pa_np, dim=1)
        a = torch.cat([a_1, a_2], dim=1)

        return a

class ContrastKnow(nn.Module):

    def __init__(self, txt_input_dim=768, img_input_dim=768):
        super(ContrastKnow, self).__init__()
        self.know_txt = nn.ParameterList([nn.Parameter(torch.randn(txt_input_dim, txt_input_dim)), nn.Parameter(torch.randn(txt_input_dim, txt_input_dim))])
        self.know_imgs = nn.ParameterList([nn.Parameter(torch.randn(img_input_dim, img_input_dim)), nn.Parameter(torch.randn(img_input_dim, img_input_dim))])
        self.tanh = nn.Tanh()
    def forward(self, imgs, texts):
        img_pos = torch.matmul(imgs, self.know_imgs[0])
        img_neg = torch.matmul(imgs, self.know_imgs[1])
        txt_pos = torch.matmul(texts, self.know_txt[0])
        txt_neg = torch.matmul(texts, self.know_txt[1])
        img_neg = self.tanh(img_neg)
        img_pos = self.tanh(img_pos)
        txt_pos = self.tanh(txt_pos)
        txt_neg = self.tanh(txt_neg)
        return img_pos, img_neg, txt_pos, txt_neg


def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    """
    计算对比损失函数

    Args:
    - embeddings1: 第一个样本的嵌入表示，shape为(batch_size, token_length, hidden_size)
    - embeddings2: 第二个样本的嵌入表示，shape为(batch_size, token_length, hidden_size)
    - labels: 标签，1表示相似对，0表示不相似对，shape为(batch_size,)
    - margin: 边界值，控制相似对和不相似对之间的距离，默认为1.0

    Returns:
    - loss: 对比损失
    """

    # 计算余弦相似度
    sim = F.cosine_similarity(embeddings1, embeddings2, dim=-1)

    # 对比损失计算
    loss = torch.mean((1 - labels) * torch.pow(sim, 2) +
                      labels * torch.pow(torch.clamp(margin - sim, min=0.0), 2))

    return loss

class KEHModel_without_know(nn.Module):
    """
    Our model for Image Repurpose Task
    """

    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768, img_inter_dim=500, img_out_dim=300,
                 cro_layers=1, cro_heads=5, cro_drop=0.2,
                 txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False,
                 img_gat_layer=2, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, img_edge_dim=0,
                 img_patch=49, lam=1, type_bmco=0, visualization=False):
        super(KEHModel_without_know, self).__init__()
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size

        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim

        if self.img_out_dim is not self.txt_out_size:
            self.img_out_dim = self.txt_out_size

        self.cro_layers = cro_layers
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.type_bmco = type_bmco

        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.img_edge_dim = img_edge_dim

        if self.img_gat_layer is not self.txt_gat_layer:
            self.img_gat_layer = self.txt_gat_layer
        if self.img_gat_drop is not self.txt_gat_drop:
            self.img_gat_drop = self.txt_gat_drop
        if self.img_gat_head is not self.txt_gat_head:
            self.img_gat_head = self.txt_gat_head

        self.img_patch = img_patch
        # 对比学习然后融合
        self.contrast = ContrastKnow(txt_input_dim=self.txt_out_size, img_input_dim=self.img_out_dim)
        # 融合
        self.tanh = nn.Tanh()
        self.linear_txt = nn.Linear(2 * self.txt_out_size, self.txt_out_size)
        self.linear_img = nn.Linear(2 * self.img_out_dim, self.img_out_dim)
        # 对比损失函数

        self.txt_encoder = TextEncoder_without_know(input_size=self.txt_input_dim, out_size=self.txt_out_size)

        self.img_encoder = ImageEncoder(input_dim=self.img_input_dim, inter_dim=self.img_inter_dim,
                                        output_dim=self.img_out_dim)
        

        self.transformers = TextDiff(self.txt_out_size, self.cro_drop)

        self.interaction = CroModality(input_size=self.img_out_dim, nhead=self.cro_heads,
                                       dim_feedforward=2 * self.img_out_dim,
                                       dropout=self.cro_drop, cro_layer=self.cro_layers, type_bmco=self.type_bmco)
        self.alignment = Alignment(input_size=self.img_out_dim, txt_gat_layer=self.txt_gat_layer,
                                   txt_gat_drop=self.txt_gat_drop,
                                   txt_gat_head=self.txt_gat_head, txt_self_loops=self.txt_self_loops,
                                   img_gat_layer=self.img_gat_layer
                                   , img_gat_drop=self.img_gat_drop, img_gat_head=self.img_gat_head,
                                   img_self_loops=self.img_self_loops,
                                   is_knowledge=0)

        self.linear1 = nn.Linear(in_features=2 * self.img_patch, out_features=2)

        self.lam = lam
        self.visulization = visualization

    def forward(self, imgs, texts, caption, mask_batch, cap_mask_batch, mask_total, img_edge_index, t1_word_seq, caption_seq,txt_edge_index,
                gnn_mask, np_mask, img_edge_attr=None, key_padding_mask_img=None):
        """
        Computes the forward pass of the network
        Args:
            imgs(N, C, W, H):  list of length N of images (C X W X H), where N denotes minibatch size,
            C, H, W denotes image channels, width and height. cpu
            texts:(N,L,D) Text embeddings of original caption. gpu
            mask_batch(N, L): Tensor. key_padding_mask for original caption. on gpu.
            img_edge_index: (N, *) Tensor. gpu
            img_edge_attr: (N,*,5) Tensor. gpu
            t1_token_length : list. must be on cpu. Input of RNN of text encoder for original caption.
            t1_word_seq:(N,) list. np seq
            txt_edge_index: gpu
            gnn_mask:(N) Boolean Tensor. gpu
            np_mask:(N,L) Boolean Tensor. gpu
        Returns:
            y: (N, 2) the similarity score of original caption and image.
        """
        imgs, pv = self.img_encoder(imgs, lam=self.lam)

        texts, score = self.txt_encoder(t1=texts, word_seq=t1_word_seq,
                                        key_padding_mask=mask_batch, lam=self.lam)
        caption, caption_saocre = self.txt_encoder(t1=caption, word_seq=caption_seq,
                                key_padding_mask=cap_mask_batch, lam=self.lam)
 
        out = self.transformers(texts, caption, mask_total)
        #img_pos, img_neg, txt_pos, txt_neg = self.contrast(imgs=imgs, texts=texts)
        #loss = contrastive_loss(img_pos, img_neg, 0) + contrastive_loss(txt_pos, txt_neg, 0)
        #texts = self.tanh(self.linear_txt(torch.cat([txt_pos, txt_neg], dim=2)))
        #imgs = self.tanh(self.linear_img(torch.cat([img_pos, img_neg], dim=2)))

        imgs, texts = self.interaction(images=imgs, texts=texts, key_padding_mask=mask_batch,
                                       key_padding_mask_img=key_padding_mask_img)

        if self.img_edge_dim == 0:
            a = self.alignment(t2=texts, v2=imgs, edge_index=txt_edge_index, gnn_mask=gnn_mask, score=score,
                               key_padding_mask=mask_batch, np_mask=np_mask, img_edge_index=img_edge_index,
                               lam=self.lam)
        else:
            # (N,49)
            a = self.alignment(t2=texts, v2=imgs, edge_index=txt_edge_index, gnn_mask=gnn_mask, score=score,
                               key_padding_mask=mask_batch, np_mask=np_mask, img_edge_index=img_edge_index,
                               img_edge_attr=img_edge_attr, lam=self.lam)

        pv = pv.repeat(1, 2)

        y = self.linear1(torch.cat([a * pv], dim=1))

        if self.visulization:
            return y, a, pv
        else:
            return y, 0




