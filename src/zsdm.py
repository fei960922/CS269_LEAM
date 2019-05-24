import pickle
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from copy import deepcopy

from utils import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.cuda.set_device(1)
    
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor        
Device = 'cuda:1' if USE_CUDA else 'cpu'

class Encoder(nn.Module):
    def __init__(self, input_size, n_latent, n_topic, 
                 dropout_rate, n_layer=1, attribute_size=50, n_hidden=256, tfidf=False):
        super().__init__()
        
        non_linearity = nn.ReLU(True)
        
        # transformer
        if not tfidf:
            self.a = nn.Sequential(nn.Linear(n_topic, attribute_size), nn.Tanh())
        
        self.a_mean = nn.Linear(attribute_size, n_latent)
        self.a_logsig = nn.Linear(attribute_size, n_latent)
        nn.init.constant_(self.a_logsig.weight, val=0)
        nn.init.constant_(self.a_logsig.bias, val=0)
        
        if dropout_rate == 0:
            layer0 = [nn.Linear(input_size, n_hidden), non_linearity]
            layer = [nn.Linear(n_hidden, n_hidden), non_linearity]
        else:
            layer0 = [nn.Linear(input_size, n_hidden), non_linearity, nn.Dropout(dropout_rate)]
            layer = [nn.Linear(n_hidden, n_hidden), non_linearity, nn.Dropout(dropout_rate)]
        
        enc = layer0+layer*(n_layer-1)
        self.enc_vec = nn.Sequential(*enc)

        self.z_mean = nn.Linear(n_hidden, n_latent)
        self.z_logsig = nn.Linear(n_hidden, n_latent)
        nn.init.constant_(self.z_logsig.weight, val=0)
        nn.init.constant_(self.z_logsig.bias, val=0)
    
    def forward(self, x, y=None, A=None):
        if x is not None:
            enc_vec = self.enc_vec(x)
            z_mean = self.z_mean(enc_vec)
            z_logsig = self.z_logsig(enc_vec)
        else:
            z_mean, z_logsig = None, None
        
        if A is None:
            A = self.a(y)
        elif y is not None:
            A = torch.mm(y, A)
           
        a_mean = self.a_mean(A)
        a_logsig = self.a_logsig(A)
        
        return z_mean, z_logsig, a_mean, a_logsig

class Decoder(nn.Module):
    def __init__(self, n_latent, output_size, dropout_rate, n_layer=1, n_hidden=256):
        super(Decoder, self).__init__()
   
        non_linearity = nn.ReLU(True)
    
        if dropout_rate == 0:
            layer0 = [nn.Linear(n_latent, n_hidden), non_linearity]
            layer = [nn.Linear(n_hidden, n_hidden), non_linearity]
        else:
            layer0 = [nn.Linear(n_latent, n_hidden), non_linearity, nn.Dropout(dropout_rate)]
            layer = [nn.Linear(n_hidden, n_hidden), non_linearity, nn.Dropout(dropout_rate)]

        dec = layer0+layer*(n_layer-1)
        self.dec_vec = nn.Sequential(*dec)
        
        self.x_mean = nn.Linear(n_hidden, output_size)
        self.x_logsig = nn.Linear(n_hidden, output_size)
        nn.init.constant_(self.x_logsig.weight, val=0)
        nn.init.constant_(self.x_logsig.bias, val=0)
                
    def forward(self, z):
        z = self.dec_vec(z)
        x_mean = torch.sigmoid(self.x_mean(z))
        # x_logsig = self.x_logsig(z)
        
        return x_mean #, x_logsig
    
class ZSDM(nn.Module):
    def __init__(self, input_size, n_topic, n_latent, attribute_size=50, dropout_rate=0.8, n_hidden=256, tfidf=False):
        super(ZSDM, self).__init__()

        # self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
    
        self.encoder = Encoder(input_size, n_latent, n_topic, dropout_rate, 
                               n_layer=1, attribute_size=50, n_hidden=n_hidden, tfidf=False) 
        self.decoder = Decoder(n_latent, input_size, dropout_rate, n_layer=1, n_hidden=n_hidden)
        
        self.w = FloatTensor(torch.eye(n_topic).to(Device))
        
        self.recon_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y, A, eps):
        z_mu, z_logsig, prior_mean, prior_logsig = self.encoder(x, y, A)
        
        # sampling
        z_samples =torch.exp(z_logsig)*eps + z_mu
    
        # decoder
        x_recon = self.decoder(z_samples)
        
        # kl loss
        kl_mat = - 0.5 * (1 - (z_mu - prior_mean)**2 
                               + 2 * (z_logsig - prior_logsig) 
                               - torch.exp(2 * (z_logsig - prior_logsig))).sum(1)
        kl_loss = kl_mat.sum()
        
        # recon
        x_recon_loss = self.recon_loss(x_recon, x)   
        val_obj = kl_loss + x_recon_loss
        
        # reg - maximum margin
        _, _, a_mean, a_logsig = self.encoder(None, self.w, A) 
        # _, _, neg_mean, neg_logsig = self.encoder(x, 1-y, A) 
        tile_mean = torch.unsqueeze(z_mu, 1)
        tile_logsig = torch.unsqueeze(z_logsig, 1)
        tile_a_mean = torch.unsqueeze(a_mean, 0)
        tile_a_logsig = torch.unsqueeze(a_logsig, 0)
        kl_mat = - 0.5 * (1 - (tile_mean - tile_a_mean)**2 
                               + 2 * (tile_logsig - tile_a_logsig)
                               - torch.exp(2 * (tile_logsig - tile_a_logsig))).sum(2)
        # margin_loss = (-kl_mat*(1-y)).logsumexp(1).sum()
        margin_loss = ((kl_mat*y).sum(1)/y.sum(1)).mean() + (-kl_mat*(1-y)).logsumexp(1).mean()
        
        return margin_loss+x_recon_loss, val_obj
    
class Trainer:
    def __init__(self, batch_size=32, num_epoches=25, n_latent=50, n_topic=5, learning_rate=1e-2, valid_freq=5, model_type='bow'):
        
        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.learning_rate = learning_rate
        self.valid_freq = valid_freq
        self.model_type = model_type
        
        self.n_latent = n_latent
        self.n_topic = n_topic
    
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
    def fit(self, x, y, class_names, A=None):

        if not hasattr(self, 'i2v'):
            raise ValueError('Must set vocabulary dict first')

        # TODO: Initialize these in the __init__()
        self.id2label = {}
        self.label2id = {}
        
        n_topic = len(class_names)
        for i, label in enumerate(class_names):
            self.id2label[str(i)] = label
            self.label2id[label] = i

        model = self.model
        # optimizer = optim.Adam(model.parameters())
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
            
            for switch in range(0, 2):                
                if switch == 0:
                    optimizer = optim.Adam(model.decoder.parameters())
                    print_mode = 'updating decoder'
                    alternate_epochs = 1
                else:
                    # rep_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
                    optimizer = optim.Adam(model.encoder.parameters())
                    print_mode = 'updating encoder'
                    alternate_epochs = 1


                for i, data in enumerate(getBatch(self.batch_size, train_data), 1):
                    if self.model_type == 'embed':
                        inputs, targets = pad_to_train(data)
                    elif self.model_type == 'bow':
                        inputs, targets = zip(*data)
                        inputs = FloatTensor(inputs)

                    model.zero_grad()
                    
                    eps = FloatTensor(inputs.size()[0], self.n_latent).normal_().to(Device)   
                    loss, val_obj = model(inputs, FloatTensor(targets), None, eps)
                    
                    losses.append(loss.data.item())
                    
                    if epoch == self.num_epoches-1:
                        train_prob.append(prob.detach())
                        if self.model_type == 'embed':
                            train_beta.append(beta.detach())

                    loss.backward()
                    optimizer.step()

            if epoch % self.valid_freq == 0:
                
                prob = self.compute_prob(x, model)
                pred, threshold, score = self.find_class(prob, y, threshold='empirical')
                match_epoch = score['match']
                hs_epoch = score['HS']
                
                prob = self.compute_prob(self.valid_x, model)
                pred, threshold, score = self.find_class(prob, self.valid_y, threshold='empirical')
                val_match = score['match']
                val_hs = score['HS']
                
                print("--- epoch:", epoch, "---")    
                print("[%d/%d] loss_epoch : %0.2f" %(epoch, self.num_epoches, np.mean(losses)),
                      "val_match : %0.4f" % val_match, "val_hs : %0.4f" % val_hs,
                      "match_epoch : %0.4f" % match_epoch, "hs_epoch : %0.4f" % hs_epoch
                    )

                if val_hs > max_val_hs:
                    max_val_hs = val_hs                        
                    self.model = deepcopy(model)

                if val_match > max_val_match:
                    max_val_match = val_match
                    self.model = deepcopy(model)

            self.model.zero_grad()
        
        print('Max_scores', max_val_match, max_val_hs)
        train_prob = self.compute_prob(x, model)
        train_pred, train_threshold, train_score = self.find_class(train_prob, y, threshold='empirical', plot=True)
        print(train_score)

        test_prob = self.compute_prob(self.valid_x, model)
        test_pred, test_threshold, test_score = self.find_class(test_prob, self.valid_y, threshold='empirical', plot=True)
        print(test_score)
        
        pred, threshold, score = self.find_class(test_prob, self.valid_y, train_threshold)     
                
        return train_prob, test_prob, train_threshold, test_threshold
    
    def compute_prob(self, x, model=None, A=None):
        if self.model_type == 'embed':
            x = pad_to_batch(x)
        elif self.model_type == 'bow':
            x = FloatTensor(x)
                
        if model == None:
            model = deepcopy(self.model)
        else:
            model = deepcopy(model)
            
        for param in model.parameters():
            param.requires_grad = False
        
        y = FloatTensor(torch.eye(self.n_topic).to(Device))
        z_mu, z_logsig, prior_mean, prior_logsig = model.encoder(x, y, A)
        
        tile_mean = torch.unsqueeze(z_mu, 1)
        tile_logsig = torch.unsqueeze(z_logsig, 1)
        tile_a_mean = torch.unsqueeze(prior_mean, 0)
        tile_a_logsig = torch.unsqueeze(prior_logsig, 0)
        kl = - 0.5 * (1 - (tile_mean - tile_a_mean)**2 
                           + 2 * (tile_logsig - tile_a_logsig)
                           - torch.exp(2 * (tile_logsig - tile_a_logsig))).sum(2)
        prob = torch.sigmoid(-kl)
        return prob
    
    def find_class(self, prob, y=None, threshold='empirical', plot=False):
        if threshold == 'empirical':
            hss, mcs, f1s = [], [], []
            thrs = np.linspace(0.11, 0.51, 200)
            for th in thrs:
                pred = multi_label_extract(prob, th)
                pred = pred.cpu().detach().numpy()
                score = multilabel_eval(y, pred, full=False)
                hss.append(score['HS'])   
                mcs.append(score['match'])   
                f1s.append(score['f1'])   
            
            idx = np.argmax(f1s)
            threshold = thrs[idx]

            if plot:
                plt.plot(thrs, hss, label='hamming score')
                plt.plot(thrs, mcs, label='match rate')
                plt.plot(thrs, f1s, label='f1')
                plt.axvline(x=threshold)
                plt.legend()
                plt.title('%.3f:%.5f' %(threshold, hss[idx]))   
                plt.show()    
                plt.clf()

        pred = multi_label_extract(prob, threshold)    
        pred = pred.cpu().detach().numpy()
        score = multilabel_eval(y, pred, full=True) 

        return pred, threshold, score

def main():
    data_dir = 'data/reuters'
    vocab_url = os.path.join(data_dir, 'vocab.pkl')
    train_url = os.path.join(data_dir, 'train')
    test_url = os.path.join(data_dir, 'test')
    vocab = pickle.load(open(vocab_url, 'rb'))
    vocab = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]
    vocab_size = len(vocab)

    train_set, train_labels, _, class_names = bow_dataset(train_url, vocab_size)
    test_set, test_labels, _, _ = bow_dataset(test_url, vocab_size)

    model = ZSDM(vocab_size, len(class_names), n_latent=128, attribute_size=50, 
                 dropout_rate=0.8, n_hidden=256, tfidf=False)
    trainer = Trainer(batch_size=128, num_epoches=200, n_latent=128, n_topic=len(class_names), 
                      learning_rate=1e-3, valid_freq=10, model_type='bow')

    trainer.set_vocab(vocab)
    trainer.set_validation(test_set, test_labels)
    trainer.init_model(model)
    train_prob, test_prob, train_threshold, test_threshold = trainer.fit(train_set, train_labels, class_names)
    
    task_url = os.path.join(data_dir, 'task')
    seq_task, seq_labels, _, _ = bow_dataset(task_url, vocab_size, monitor=False)
    test_probs = trainer.compute_prob(seq_task)
    test_pred, threshold, score = trainer.find_class(test_probs, seq_labels, threshold=test_threshold)
    print(score)    