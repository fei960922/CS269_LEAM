import random
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy

from torch import nn, optim, tensor
from torch.autograd import Variable
from src.utils import *
from src.model import *

#print(USE_CUDA = torch.cuda.is_available())

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
        data_iterator,
        batch_size=32,
        num_epoches=25,
        learning_rate=1e-2,
        valid_freq=5,
        model_type='embed',
        
    ): # bow

        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.valid_freq = valid_freq
        self.model_type = model_type
        self.iterator = data_iterator 
        self.train_set = None
    
    
    # TODO: Populate the docstring.
    def set_vocab(self, vocab):
        if len(vocab) > 0:
            self.i2v = vocab

    # TODO: Populate the docstring.
    def set_validation(self, val_set):
        self.val_x = []
        self.val_y = []
        num_val = self.iterator.get_num_batches(val_set)
        for i in range(num_val):
            batch = next(iter(self.iterator(val_set)))
            self.val_x.append(batch["tokens"])
            self.val_y.append(batch["label"].type(ByteTensor))
        self.val_y = torch.cat(self.val_y, 0)

    def init_model(self, model):
        if USE_CUDA:
            model = model.cuda()
        self.model = model
        return self.model

    # TODO: Populate the docstring.
    def fit(self, data_set, class_names):
        '''
        if not hasattr(self, 'i2v'):
            raise ValueError('Must set vocabulary dict first')
        '''
        # TODO: Initialize these in the __init__()
        self.id2label = {}
        self.label2id = {}
        for i, label in enumerate(class_names):
            self.id2label[str(i)] = label
            self.label2id[label] = i

        model = self.model
        optimizer = optim.Adam(model.parameters())
        max_val_match = 0.
        max_val_hs = 0.
        self.train_set = data_set

        for epoch in range(self.num_epoches):
            losses = []
            match = []
            hs = []
            if epoch == self.num_epoches-1:
                
                train_prob = []
                if self.model_type == 'embed':
                    train_beta = []
            num_iter = self.iterator.get_num_batches(data_set) 
            for i in range(num_iter):
                batch = next(iter(self.iterator(data_set)))
                inputs = batch["tokens"]
                targets = batch["label"]
                
                
                model.zero_grad()
                prob, beta, loss = model(inputs, targets.type(FloatTensor))
                
                
                if epoch == self.num_epoches-1:
                    train_prob.append(prob.detach())
                    if self.model_type == 'embed':
                        train_beta.append(np.squeeze(beta.detach().cpu().numpy()))
                
                losses.append(loss.data.item())
                
                pred = prob>0.5
                true = targets.type(ByteTensor)
                match += [(pred[i][true[i]==1]==1).any().float() for i in range(len(pred))]
                hs += [(((pred==1)*(true==1)).sum(1)/(((pred==1)+(true==1))>0).sum(1)).float()]
   
                loss.backward()
                optimizer.step()


            if epoch % self.valid_freq == 0 or epoch == self.num_epoches-1:
                match_epoch = torch.mean(torch.stack(match))
                hs_epoch = torch.mean(torch.cat(hs))
                
                prob, _ = self.predict(model)
                pred = prob>0.5
                #print(np.sum(pred, axis=-1))
            
                
                tmp = self.val_y.detach().cpu().numpy()
                #print(np.sum(tmp, axis=-1)[0:30])
                #print(np.sum((pred==1)*(tmp==1), axis=-1)[0:30])
                #print(np.sum(((pred==1) + (tmp==1))>0, axis=-1)[0:30])

                val_match = np.mean([(pred[i][tmp[i]==1]==1).any() for i in range(len(pred))])
                val_hs = (((pred==1)*(tmp==1)).sum(1)/(((pred==1)+(tmp==1))>0).sum(1)).mean()
                
                print("--- epoch:", epoch, "---")    
                print("[%d/%d] loss_epoch : %0.2f" %(epoch, self.num_epoches, np.mean(losses)),
                      "val_match : %0.4f" % val_match, "match_epoch : %0.4f" % match_epoch, 
                      "val_hs : %0.4f" % val_hs,
                      "hs_epoch : %0.4f" % hs_epoch
                    ) 

                if val_hs >= max_val_hs:
                    max_val_hs = val_hs                        
                    self.model = deepcopy(model)

                if val_match >= max_val_match:
                    max_val_match = val_match
                    self.model = deepcopy(model)
                    
            self.model.zero_grad()
        
        print('Max_scores', max_val_match, max_val_hs)
        train_score = self.evaluate(train_score=True)
        val_score = self.evaluate(train_score=False)
        print('training', train_score)
        print('validation', val_score)
        
        train_prob = np.array(torch.cat(train_prob, 0).tolist())
        if self.model_type == 'embed':
            longest = 0
            betas_same_length = []
            for beta in train_beta:
                length = beta.shape[-1]
                longest = max(longest, length)
            print(longest)    
            for beta in train_beta:
                tmp = np.concatenate([beta, np.zeros((beta.shape[0], longest - beta.shape[1]))], axis=1)
                betas_same_length.append(tmp)
            betas_same_length = np.concatenate(betas_same_length, axis=0)
            
            return train_prob, betas_same_length
        return train_prob
    
    def predict(self, model=None):
        if model == None:
            model = self.model
        model.eval() 
        '''
        for param in model.parameters():
            param.requires_grad = False
        '''
        probs, betas = [], []
        for i in range(len(self.val_x)):
            data = self.val_x[i]
            
            prob, beta = model(data)
            if self.model_type == 'embed':
                betas.append(np.squeeze(beta.detach().cpu().numpy()))
            probs.append(prob)
                                       
        probs = np.array(torch.cat(probs, 0).tolist())
        betas_same_length = []
        if self.model_type == 'embed':
            longest = 0
            
            for beta in betas:
                length = beta.shape[-1]
                longest = max(longest, length)
            for beta in betas:
                tmp = np.concatenate([beta, np.zeros((beta.shape[0], longest - beta.shape[1]))], axis=1)
                betas_same_length.append(tmp)
            betas_same_length = np.concatenate(betas_same_length, axis=0)
                
        model.train()
        return probs, betas_same_length
    
    #To do, this function need to be changed
    def evaluate(self, train_score=False, model=None):
        if model == None:
            model = self.model
        
        model.eval()
        if train_score == True:
            assert self.train_set is not None
            prob = []
            y = []
            num_iter = self.iterator.get_num_batches(self.train_set) 
            for i in range(num_iter):
                batch = next(iter(self.iterator(self.train_set)))
                x = batch["tokens"]
                targets = batch["label"].type(ByteTensor)
                tmpprob, _ = model(x)
                prob.append(tmpprob)
                y.append(targets)
            prob = torch.cat(prob, 0).detach().cpu().numpy()
            y = torch.cat(y, 0)
        else:
            prob, _ = self.predict(model)
            y = self.val_y
            
        
        pred = prob > 0.5
        
        score = multilabel_eval(y.detach().cpu().numpy(), pred, full=True)
        model.train()
        return score
