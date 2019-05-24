import random
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy

from torch import nn, optim, tensor
from torch.autograd import Variable
from utils import *
from model import *

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
        
class Trainer:
    def __init__(
        self,
        batch_size=32,
        num_epoches=25,
        learning_rate=1e-2,
        valid_freq=5,
        model_type='embed' 
    ): # bow
        
        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.valid_freq = valid_freq
        self.model_type = model_type
    
    # TODO: Populate the docstring.
    def set_vocab(self, vocab):
        if len(vocab) > 0:
            self.i2v = vocab

    # TODO: Populate the docstring.
    def set_validation(self, valid_x, valid_y):
        self.valid_x = valid_x
        self.valid_y = valid_y

    def init_model(self, model):
        if USE_CUDA:
            model = model.cuda()
        self.model = model
        return self.model

    # TODO: Populate the docstring.
    def fit(self, x, y, class_names):

        if not hasattr(self, 'i2v'):
            raise ValueError('Must set vocabulary dict first')

        # TODO: Initialize these in the __init__()
        self.id2label = {}
        self.label2id = {}

        for i, label in enumerate(class_names):
            self.id2label[str(i)] = label
            self.label2id[label] = i

        model = self.model
        optimizer = optim.Adam(model.parameters())
        train_data = list(zip(x, y))
        max_val_match = 0.
        max_val_hs = 0.

        for epoch in range(self.num_epoches):
            losses = []
            match = []
            hs = []
            if epoch == self.num_epoches-1:
                train_prob = []
                if self.model_type == 'embed':
                    train_beta = []

            for i, data in enumerate(getBatch(self.batch_size, train_data), 1):
                if self.model_type == 'embed':
                    inputs, targets = pad_to_train(data)
                elif self.model_type == 'bow':
                    inputs, targets = zip(*data)
                
                model.zero_grad()
                if self.model_type == 'embed':
                    prob, beta, loss = model(inputs, FloatTensor(targets))
                elif self.model_type == 'bow':
                    prob, loss = model(FloatTensor(inputs), FloatTensor(targets))
                
                if epoch == self.num_epoches-1:
                    train_prob.append(prob.detach())
                    if self.model_type == 'embed':
                        train_beta.append(beta.detach())
                
                losses.append(loss.data.item())
                
                pred = prob>0.5
                true = ByteTensor(targets)
                match += [(pred[i][true[i]]==1).any().float() for i in range(len(pred))]
                hs += [(((pred==1)*(true==1)).sum(1)/(((pred==1)+(true==1))>0).sum(1)).float()]

                loss.backward()
                optimizer.step()

            if epoch % self.valid_freq == 0:
                match_epoch = torch.mean(torch.stack(match))
                hs_epoch = torch.mean(torch.cat(hs))
                
                prob, _ = self.predict(self.valid_x)
                pred = prob>0.5
                val_match = np.mean([(pred[i][self.valid_y[i]]==1).any() for i in range(len(pred))])
                val_hs = (((pred==1)*(self.valid_y==1)).sum(1)/(((pred==1)+(self.valid_y==1))>0).sum(1)).mean()
                
                print("--- epoch:", epoch, "---")    
                print("[%d/%d] loss_epoch : %0.2f" %(epoch, self.num_epoches, np.mean(losses)),
                      "val_match : %0.4f" % val_match, "match_epoch : %0.4f" % match_epoch, 
                      "val_hs : %0.4f" % val_hs,
                      "hs_epoch : %0.4f" % hs_epoch
                    ) 

                if val_hs > max_val_hs:
                    max_val_hs = val_hs                        
                    self.model = deepcopy(model)

                if val_match > max_val_match:
                    max_val_match = val_match
                    self.model = deepcopy(model)

            self.model.zero_grad()
        
        print('Max_scores', max_val_match, max_val_hs)
        train_score = self.evaluate(x, y)
        val_score = self.evaluate(self.valid_x, self.valid_y)
        print('training', train_score)
        print('validation', val_score)
        
        train_prob = np.array(torch.cat(train_prob, 0).tolist())
        if self.model_type == 'embed':
            train_beta = np.array(torch.cat(train_beta, 0).tolist())
            return train_prob, train_beta
        return train_prob
    
    def predict(self, x, model=None):
        if model == None:
            model = deepcopy(self.model)
            
        for param in model.parameters():
            param.requires_grad = False
        
        probs, betas = [], []
        for i, data in enumerate(getBatch(self.batch_size, x)):
            if self.model_type == 'embed':
                data = pad_to_batch(data)
                prob, beta = model(data)
                betas.append(beta)
                
            elif self.model_type == 'bow':
                data = FloatTensor(data)
                prob = model(data)

            probs.append(prob)
                                       
        probs = np.array(torch.cat(probs, 0).tolist())
        
        if self.model_type == 'embed':
            betas = np.array(torch.cat(betas, 0).tolist())
        
        return probs, betas
    
    def evaluate(self, x, y, model=None):
        if model == None:
            model = self.model
        
        prob, _ = self.predict(x, model)
        pred = prob > 0.5
        
        score = multilabel_eval(y, pred, full=True)

        return score