# -*- coding: utf-8 -*-
# @Time    : 2021/05/01
# @Author  : Xinyan Fan
# @Email   : xinyan.fan@ruc.edu.cn

"""
LightSANs
################################################
# code modified by jhpark for final project on SKKU : Recommender Systems 
################################################

Reference:
    Xin-Yan Fan et al. "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." in SIGIR 2021.



Reference:
    https://github.com/BELIEVEfxy/LightSANs
"""



"""
2024. 12. 11.
code modified by jhpark
SKKU Recommender System Final Project 

implementation for dynamic k selection
"""



import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import os

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import LightTransformerEncoder

class LightSANs(SequentialRecommender):

    def __init__(self, config, dataset):
        super(LightSANs, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.k_interests = config['k_interests']
        self.hidden_size = config['hidden_size'] # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.seq_len = self.max_seq_length
        
        # RecBole 기본 설정에서 USER_ID_FIELD를 통해 사용자 ID 필드명 획득
        self.USER_ID = config['USER_ID_FIELD']  

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size , padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = LightTransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads, 
                                              k_interests=self.k_interests, hidden_size=self.hidden_size, 
                                              seq_len=self.seq_len,
                                              inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # user_complexities: {user_id: [(variance, entropy), ...]}
        self.user_complexities = defaultdict(list)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def embedding_layer(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        return item_emb, position_embedding

    def _calculate_complexities(self, item_seq, item_emb):
        # item_emb: [B, N, d]
        # Variance complexity
        mean_emb = item_emb.mean(dim=1, keepdim=True)  # [B,1,d]
        variance_complexity = ((item_emb - mean_emb)**2).mean(dim=(1,2))  # [B]

        # Entropy complexity based on item distribution
        B, N = item_seq.shape
        entropy_list = []
        for i in range(B):
            seq_np = item_seq[i][item_seq[i]!=0] # remove padding
            if len(seq_np) == 0:
                entropy_list.append(0.0)
                continue
            unique_items, counts = torch.unique(seq_np, return_counts=True)
            p = counts.float() / counts.sum()
            entropy = - (p * torch.log(p + 1e-9)).sum()
            entropy_list.append(entropy.item())
        entropy_complexity = torch.tensor(entropy_list, device=item_seq.device)

        return variance_complexity, entropy_complexity

    def forward(self, item_seq, item_seq_len):
        item_emb, position_embedding = self.embedding_layer(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)

        variance_complexity, entropy_complexity = self._calculate_complexities(item_seq, item_emb)
        #complexity_score = variance_complexity + (0.1 * entropy_complexity)
        complexity_score = entropy_complexity
        
        # trm_encoder가 (all_encoder_layers, g) 반환
        trm_output, g = self.trm_encoder(item_emb,
                                        position_embedding,
                                        complexity_score=complexity_score,
                                        output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output, variance_complexity, entropy_complexity, g

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]  # [B]

        seq_output, variance_complexity, entropy_complexity, g = self.forward(item_seq, item_seq_len)


        # user_complexities 기록
        for uid, var_c, ent_c, user_g in zip(user_id.tolist(),
                                            variance_complexity.tolist(),
                                            entropy_complexity.tolist(),
                                            g.detach().cpu().numpy()):
            self.user_complexities[uid].append((var_c, ent_c, user_g))

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output, variance_complexity, entropy_complexity = self.forward(item_seq, item_seq_len)
        # predict에서는 user_complexities를 기록하지 않음

        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # 여기서 variance_complexity, entropy_complexity, g는 사용하지 않으므로 _로 대체
        seq_output, _, _, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def save_user_complexities(self, save_path='user_complexities.pt'):
        # 학습 종료 후 이 함수 호출 시 user_complexities를 저장
        torch.save(dict(self.user_complexities), save_path)
        print(f"User complexities saved to {save_path}")
