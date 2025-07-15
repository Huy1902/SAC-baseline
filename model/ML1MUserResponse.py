from model.BaseModel import BaseModel
import torch.nn as nn
import torch
from model.DNN import DNN
class ML1MUserResponse(BaseModel):
  def __init__(self, reader, params):
    super().__init__(reader, params)
    self.bce_loss = nn.BCEWithLogitsLoss(reduction= 'none')

  def _define_params(self, reader, params):
    stats = reader.get_statistics()
    print(stats)
    self.potrait_len = stats['user_portrait_len']
    self.item_dim = stats['item_vec_size']
    self.feature_dim = params['feature_dim']
    self.hidden_dim = params['hidden_dims']
    self.attn_n_head = params['attn_n_head']
    self.dropout_rate = params['dropout_rate']
    self.uEmb = nn.Embedding.from_pretrained(torch.FloatTensor(self.reader.user_meta), freeze=False)
    self.iEmb = nn.Embedding.from_pretrained(torch.FloatTensor(self.reader.item_meta), freeze=False)

    # fuse information
    self.concat_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

    # portrait embedding
    self.portrait_encoding_layer = DNN(self.potrait_len, self.hidden_dim,
                                        self.feature_dim, self.dropout_rate,
                                        do_batch_norm= False)
    # item embedding
    self.item_emb_layer = nn.Linear(self.item_dim, self.feature_dim)

    # user history encoder
    self.seq_self_attn_layer = nn.MultiheadAttention(self.feature_dim, self.attn_n_head, batch_first= True)
    self.seq_user_attn_layer = nn.MultiheadAttention(self.feature_dim, self.attn_n_head, batch_first= True)

    self.loss = []

  def get_forward(self, feed_dict: dict) -> dict:
    user_emb = self.portrait_encoding_layer(feed_dict['user_profile']).view(-1, 1, self.feature_dim)
    history_item_emb = self.item_emb_layer(feed_dict['history_features'])

    seq_encoding, attn_weight = self.seq_self_attn_layer(history_item_emb, history_item_emb, history_item_emb)

    user_interest, attn_weight = self.seq_user_attn_layer(user_emb, seq_encoding, seq_encoding)

    user_interest = torch.concat([user_interest, user_emb], axis=-1)
    user_interest = self.concat_layer(user_interest)

    exposure_item_emb = self.item_emb_layer(feed_dict['exposure_features'])

    score = torch.sum(exposure_item_emb * user_interest, dim=-1)

    # regularization
    reg = self.get_regularization(self.uEmb, self.iEmb, self.portrait_encoding_layer,
                                  self.item_emb_layer, self.seq_user_attn_layer,
                                  self.seq_self_attn_layer)
    return {'preds': score, 'reg': reg}

  def get_loss(self, feed_dict: dict, out_dict: dict):
    preds, reg = out_dict["preds"].view(-1), out_dict["reg"]
    target = feed_dict['feedback'].view(-1).to(torch.float)

    # print(f"preds: {self.sigmoid(preds)}")
    # print(f"target: ", target)

    loss = torch.mean(self.bce_loss(self.sigmoid(preds), target))
    # print(f"loss: {loss} l2: {reg} l2*coef: {self.l2_coef * reg}")
    self.loss.append(loss.item())
    loss = loss + self.l2_coef * reg
    return loss
