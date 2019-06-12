import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from src.utils import *
from src.trainer import *

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import util as nn_util


USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.cuda.set_device(0)
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor   

class Embedding(nn.Module):
    def __init__(self, n_topic, embedding_dim, ngram, dropout_rate, label_att=False):
        super().__init__()
        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
        
        
        self.embedding_class = nn.Embedding(n_topic, embedding_dim)
        self.conv = torch.nn.Conv1d(n_topic, n_topic, 2*ngram+1, padding=ngram)
        self.n_topic = n_topic
        self.label_att = label_att
            
    def forward(self, inputs):
        # (B, L, e)
        inputs = nn_util.move_to_device(inputs, 0 if USE_CUDA else -1)
        emb = self.word_embeddings(inputs)
        
        if self.label_att:
            emb_norm = F.normalize(emb, p=2, dim=2, eps=1e-12)
            #print(emb_norm.shape)
            emb_c = self.embedding_class(LongTensor(
                [[i for i in range(self.n_topic)] for j in range(emb.size(0))]
            ))

            emb_c_norm = F.normalize(emb_c, p=2, dim=2, eps=1e-12)
            #print(emb_c_norm.shape)
            # (B, e, L)
            emb_norm_t = emb_norm.permute(0, 2, 1)
            #print(emb_norm_t.shape)
            # (B, C, L)
            g = torch.bmm(emb_c_norm, emb_norm_t)
            
            g = F.relu(self.conv(g))
            # (B, L, 1)]
            beta = torch.max(g, 1)[0].unsqueeze(2)
            # (B, L, 1)
            beta = F.softmax(beta, 1)
            # (B, L, 1)*(B, L, e)
            z = torch.mul(beta, emb)
            # (B, e)
            output = z.sum(1)            
        else:
            output = emb.mean(1)
            beta = torch.ones(emb.size()[:2])/len(emb)
        
        return output, beta       

class Classifier(nn.Module):
    def __init__(self, n_feature, n_topic, n_layer=2, n_hidden=256, dropout_rate=0, multilabel=True):
        super().__init__()

        non_linearity = nn.ReLU(True) # nn.Softmax(), nn.Tanh()
        
        if dropout_rate == 0:
            layer0 = [nn.Linear(n_feature, n_hidden), non_linearity]
            layer = [nn.Linear(n_hidden, n_hidden), non_linearity]
        else:
            layer0 = [nn.Linear(n_feature, n_hidden), non_linearity, nn.Dropout(dropout_rate)]
            layer = [nn.Linear(n_hidden, n_hidden), non_linearity, nn.Dropout(dropout_rate)]
        
        mlp = layer0+layer*(n_layer-1)+[nn.Linear(n_hidden, n_topic)]
        
        self.classifier = nn.Sequential(*mlp)
        
        self.multilabel_loss = nn.BCEWithLogitsLoss()
        self.multiclass_loss = nn.CrossEntropyLoss()   
        self.focal_loss = FocalLoss(class_num=n_topic, alpha=None, gamma=1, size_average=True)
        
        self.multilabel = multilabel
            
    def forward(self, feature, target=None):
        output = self.classifier(feature.float())
        if self.multilabel:
            prob = torch.sigmoid(output)
        else:
            prob = torch.softmax(output, 1)
        
        if target is not None:
            if self.multilabel:
                loss = self.multilabel_loss(output, target) # self.focal_loss(output, target) #self.multilabel_loss(output, target)
            else:
                logit = F.log_softmax(output, 0.5)
                loss = self.multiclass_loss(logit, target)
                
            return prob, loss        
        return prob           

class Leam_Classifier(nn.Module):
    def __init__(self, n_topic, embeding_dim, n_hidden, ngram, 
                 n_layer, dropout_rate, embpath=None, label_att=True, multilabel=True):    
        super().__init__()
        
        self.embedding = Embedding(n_topic, embeding_dim, ngram, dropout_rate, label_att)
        self.classifier = Classifier(embeding_dim, n_topic, n_layer, n_hidden, dropout_rate, multilabel)
    
    def forward(self, inputs, target=None):
        embed, beta = self.embedding(inputs)
        if target is not None:
            prob, loss = self.classifier(embed, target)
            return prob, beta, loss        
        
        prob = self.classifier(embed, target)     
        return prob, beta               
