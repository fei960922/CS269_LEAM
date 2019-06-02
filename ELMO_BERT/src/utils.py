
'''
Functions for use
'''
import torch
from torch.autograd import Variable

import numpy as np
import itertools
import os
import random
import pickle
import torch as t

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

from sklearn.feature_extraction.text import TfidfTransformer
# from tsne import tsne

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.cuda.set_device(0)
    
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor        
Device = 'cuda:1' if USE_CUDA else 'cpu'


"""process data input to bag-of-words representation"""
def dataset(data_url, monitor=False): 
    data_withorder = np.load(data_url + '.multi.npy', allow_pickle=True)
    seq = [list(x) for x in data_withorder]
    if monitor:
        print('Converting data to sequence')
    
    try:
        labels_with_names = np.load(data_url + '.labels.multi.npy', allow_pickle=True)
        labels = labels_with_names[0]
        class_names = labels_with_names[1]
    except:
        if monitor:
            print("No labels.")
        labels = None
        class_names = ['None']
    
    return seq, labels, class_names

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)
def bow_dataset(data_url, vocab_size, additional_text=False, monitor=False):
    data_withorder = np.load(data_url + '.multi.npy', allow_pickle=True)
    if monitor:
        print('Converting data to BoW representation')
    data_multihot = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_withorder])
    word_count = [np.sum(doc) for doc in data_multihot]
    try:
        labels_with_names = np.load(data_url + '.labels.multi.npy', allow_pickle=True)
        labels = labels_with_names[0]
        class_names = labels_with_names[1]
    except:
        if monitor:
            print("No labels.")
        labels = None
        class_names = ['None']

    if additional_text:
        return data_multihot, labels, word_count, class_names, data_withorder
    
    return data_multihot, labels, word_count, class_names

'''Create batches'''
def pad_to_batch(x, max_len=80):
    x_p = []
    for i in range(len(x)):
        x_len = len(x[i])

        if x_len < max_len:
            x_p.append(Variable(LongTensor(x[i] + [0]*(max_len - x_len))))
        else:
            x_p.append(Variable(LongTensor(x[i][:max_len])))

    return torch.stack(x_p, 0)

# TODO: Populate the docstring.
def pad_to_train(batch, max_len=80):
    x, y = zip(*batch)
    return pad_to_batch(x, max_len=max_len), y

def getBatch(batch_size, train_data, shuffle=False):
    if shuffle:
        random.shuffle(train_data)

    sindex = 0
    eindex = batch_size

    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def getBatch_iter(batch_size, train_data, shuffle=False):
    if shuffle:
        random.shuffle(train_data)
    ret = []

    while True:
        for i, data in enumerate(train_data):
            ret.append(data)

            if i % batch_size == 0:
                yield ret
                ret = []

        if len(ret) > 0:
            yield ret

        break
        
'''Build attributes'''
def build_A(data_url, vocab_size, n_attribute):
    data, labels, _, _ = bow_dataset(data_url, vocab_size)
    n_label = labels.shape[1]
    n_vocab = len(data[1])
    A_large = np.zeros([n_label, n_vocab])
    for i, doc in enumerate(data):
        A_large[labels[i]==1] += doc 
    
    transformer = TfidfTransformer(smooth_idf=False)
    A_tfidf = transformer.fit_transform(A_large).toarray()
    A = tsne.tsne(A_tfidf, n_attribute, n_vocab)
    return FloatTensor(A)


'''Extract labels based on probability'''
def multi_label_extract(label_dist, threshold=0.5):
    labels = torch.zeros(label_dist.size())
    labels[label_dist > threshold] = 1
    return labels

def plot_threshold(thrs, var, threshold, title, savedir=None):
    for v in var:
        plt.plot(thrs, v[0], label=v[1])
    plt.axvline(x=threshold)
    plt.legend()
    plt.title(title)  
    if savedir is not None:
        plt.savefig('%s_%s.png'%(savedir, title))
    plt.show()    
    plt.clf()

'''Evaluation'''
# Sorower, Mohammad S. "A literature survey on algorithms for multi-label learning." Oregon State University, Corvallis (2010)
def multilabel_eval(true, pred, sample_weight=None, monitor=False, full=False):
    n, p = true.shape
    score = {}
    score['match'] = np.mean([(pred[i][true[i]==1]==1).any() for i in range(len(pred))])
    hit = ((pred==1)*(true==1)).sum(1)
    score['HS'] = (hit/(((pred==1)+(true==1))>0).sum(1)).mean()
    score['f1'] = (2*hit/((pred==1).sum(1)+(true==1).sum(1))).mean()
    
    if full:
        match = (pred==true)
        score['HL'] = (pred!=true).mean(1).mean()
        score['exact_acc'] = match.all(1).mean()
        score['min_acc'] = match.mean(0).min()
        score['density_chosen'] = pred.sum(1).mean()/p
        score['density'] = true.sum(1).mean()/p
        score['precision'] = (hit/(true==1).sum(1)).mean()
        score['recal'] = (hit/((pred==1).sum(1)+1e-12)).mean()
        score['no_pred'] = (pred!=1).all(1).mean()
        
    if monitor:
        print(score)
        
    return score

def singlelabel_eval(true, pred, sample_weight=None, monitor=False):
    score = {}
    score['acc'] = accuracy_score(true, pred)
    score['precision'] = precision_score(true, pred)
    score['recal'] = recall_score(true, pred)
    score['f1'] = f1_score(true, pred)
    score['cfm'] = confusion_matrix(true, pred)    
    if monitor:
        print('Acc: %5f, F1: %5f, precision: %5f, recall: %5f' %(score['acc'], score['f1'], score['precision'], score['recal']))
    return score

def inference_analysis(class_word, vocab_url, class_names):
    if type(topic_word) is not np.ndarray:
        topic_word = topic_word.data.cpu().numpy()
        
    if 'pkl' in vocab_url:
        vocab = pickle.load(open(vocab_url, 'rb'))
        vocab = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]
    else:
        vocab = []
        with open(vocab_url, 'r') as fin:
            for line in fin:
                vocab.append(line.split(' ')[0])
    
    for i, weights in enumerate(topic_word):
        ind = np.argsort(topic)[-1:-21:-1]
        if len(names) == len(topic_word):
            print(names[i])
        print(np.array(vocab)[ind])

def save_res_multi(tests, vals, trains, class_names, vocab_url):
    if 'pkl' in vocab_url:
        vocab = pickle.load(open(vocab_url, 'rb'))
        vocab = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]
    else:
        vocab = []
        with open(vocab_url, 'r') as fin:
            for line in fin:
                vocab.append(line.split(' ')[0]) 
    vocab_size = len(vocab)
    
    import csv
    with open('res.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['sentence', 'true_label', 'prediction', 'word_by_importance', 'dataset'])
        for pred, sent, recon in tests:
            sent_order = ' '.join([vocab[word] for word in sent])
            true_label = ' '
            pred_labels = '+'.join(class_names[pred==1])
            recon_sent = np.argsort(recon)[::-1]
            sent_importance = ' '.join([vocab[word] for word in recon_sent if word in sent])
            group = 'test'
            spamwriter.writerow([sent_order, true_label, pred_labels, sent_importance, group])
        for pred, true, sent, recon in vals:
            pred_labels = '+'.join(class_names[pred==1])
            true_label = '+'.join(class_names[true==1])
            sent_order = ' '.join([vocab[word] for word in sent])
            recon_sent = np.argsort(recon)[::-1]
            sent_importance = ' '.join([vocab[word] for word in recon_sent if word in sent])
            group = 'validation'
            spamwriter.writerow([sent_order, true_label, pred_labels, sent_importance, group])
        for pred, true, sent, recon in trains:
            pred_labels = '+'.join(class_names[pred==1])
            true_label = '+'.join(class_names[true==1])
            sent_order = ' '.join([vocab[word] for word in sent])
            recon_sent = np.argsort(recon)[::-1]
            sent_importance = ' '.join([vocab[word] for word in recon_sent if word in sent])
            group = 'train'
            spamwriter.writerow([sent_order, true_label, pred_labels, sent_importance, group])
    
    print('Result saved in csv.')      
        
        
def print_res_multi(tests, vals, trains, class_names, topic_word, class_word, vocab_url):    
    if 'pkl' in vocab_url:
        vocab = pickle.load(open(vocab_url, 'rb'))
        vocab = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]
    else:
        with open(vocab_url, 'r') as fin:
            for line in fin:
                vocab.append(line.split(' ')[0]) 
    vocab_size = len(vocab)
    
    with open('res.html', 'w', encoding='gbk') as f:
        if topic_word is not None:
            f.write('<p style="background-color:green;">Topic word (beta)</p>')
            for i, topic in enumerate(topic_word):
                ind = np.argsort(topic)[-1:-21:-1]
                f.write('<p> {} </p>'.format(class_names[i]))
                for word in ind:
                    f.write('{} '.format(vocab[word]))
                f.write('</p>')
        if class_word is not None:
            f.write('<p style="background-color:green;">Class word (sum_theta*beta)</p>')
            for i, topic in enumerate(class_word):
                ind = np.argsort(topic)[-1:-21:-1]
                f.write('<p> {} </p>'.format(class_names[i]))
                for word in ind:
                    f.write('{} '.format(vocab[word]))
                f.write('</p>')
       
        f.write('<p style="background-color:green;">Test</p>')
        for pred_val, sent, recon in tests:
            f.write('<p>validation threshold: {}, train threshold: {}</p>'.format(class_names[pred_val==1], class_names[pred_train==1]))
            f.write('<p>')

            f.write('<p> In Order:')
            for word in sent:
                f.write('{} '.format(vocab[word]))
            f.write('</p>')

            f.write('<p> By Importance:')
            recon_sent = np.argsort(recon)[::-1]
            for word in recon_sent:
                if word in sent:
                    f.write('{} '.format(vocab[word]))
            f.write('</p>')

            f.write('<p> Reconstruct:')
            for word in recon_sent:
                if recon[word]>=1/vocab_size*10:
                    if word in sent:
                        f.write('<mark class="red">{}</mark> '.format(vocab[word]))
                    else:
                        f.write('{} '.format(vocab[word]))
                else:
                    break
            f.write('</p>') 
            f.write('<HR SIZE=5>')

        if vals is not None:
            f.write('<p style="background-color:green;">Validation</p>')
            for pred, true, sent, recon in trains:
                if (pred[true==1] != 1).all():
                    f.write('<p style="background-color:red;">All Wrong</p>')
                elif (pred != true).any():
                    f.write('<p style="background-color:blue;">Partial Wrong</p>')                    
                f.write('<p>prediction: {}, true: {}</p>'.format(class_names[pred==1], class_names[true==1]))
                f.write('<p>')

                f.write('<p> In Order:')
                for word in sent:
                    f.write('{} '.format(vocab[word]))
                f.write('</p>')

                f.write('<p> By Importance:')
                recon_sent = np.argsort(recon)[::-1]
                for word in recon_sent:
                    if word in sent:
                        f.write('{} '.format(vocab[word]))
                f.write('</p>')

                f.write('<p> Reconstruct:')
                for word in recon_sent:
                    if recon[word]>=1/vocab_size*10:
                        if word in sent:
                            f.write('<mark class="red">{}</mark> '.format(vocab[word]))
                        else:
                            f.write('{} '.format(vocab[word]))
                    else:
                        break
                f.write('</p>') 
                f.write('<HR SIZE=5>')
           
        f.write('<p style="background-color:green;">Train</p>')
        for pred, true, sent, recon in trains:
            if (pred[true==1] != 1).all():
                f.write('<p style="background-color:red;">All Wrong</p>')
            elif (pred != true).any():
                f.write('<p style="background-color:blue;">Partial Wrong</p>')                    
            f.write('<p>prediction: {}, true: {}</p>'.format(class_names[pred==1], class_names[true==1]))
            f.write('<p>')

            f.write('<p> In Order:')
            for word in sent:
                f.write('{} '.format(vocab[word]))
            f.write('</p>')
            
            f.write('<p> By Importance:')
            recon_sent = np.argsort(recon)[::-1]
            for word in recon_sent:
                if word in sent:
                    f.write('{} '.format(vocab[word]))
            f.write('</p>')
            
            f.write('<p> Reconstruct:')
            for word in recon_sent:
                if recon[word]>=1/vocab_size*10:
                    if word in sent:
                        f.write('<mark class="red">{}</mark> '.format(vocab[word]))
                    else:
                        f.write('{} '.format(vocab[word]))
                else:
                    break
            f.write('</p>') 
            f.write('<HR SIZE=5>')
        
        print('Result saved in html.')
        
'''Visualization for development'''
def plot_training(caches, labels, rec, names, save=False):
    n = len(names)
    plt.figure(figsize=(5*n, n))
    plt.clf()
    gs = gridspec.GridSpec(1, n)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(n):
        plt.subplot(gs[i])
        title = '%s_Plot' %(names[i])
        plt.title(title)
        plt.xlabel('Training Steps')
        plt.ylabel(names[i])
        for j, values in enumerate(caches[i]):
            plt.plot(rec[i], values, label=labels[i][j])
    if save:
        plt.savefig('fig/log.png')
    plt.show()

    
def multilabel_confusion_matrix(true, pred, labels, normalize=False, cmap=plt.cm.Blues):
    from sklearn.metrics import confusion_matrix

    conf_mats=[]

    for label_col in range(len(labels)):
        true_label = true[:, label_col]
        pred_label = pred[:, label_col]
        conf_mats.append(confusion_matrix(pred_label, true_label))
        
    plt.figure(figsize=(5*len(labels), len(labels)))
    plt.clf()
    gs = gridspec.GridSpec(1, len(labels))
    gs.update(wspace=1./len(labels), hspace=1./len(labels))
    for i, label in enumerate(labels):
        if normalize:
            cm = conf_mats[i].astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm = conf_mats[i]
        plt.subplot(gs[i])
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(label)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].astype('float')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def data_visualize_multilabel(data, labels, n_topic, classnames, save=False, show=True, legend=True):
    idx = np.random.choice(range(len(labels)), 100, replace=False)
    transformer = TfidfTransformer(smooth_idf=False)
    data = transformer.fit_transform(data[idx]).toarray()
    labels = np.array(labels)[idx]
    Y = tsne.tsne(data, 2, data.shape[0])
    
    colors = ['orange'] + ['blue'] + ['y'] + ['m'] + ['r']
    for i in range(n_topic):
        idx = np.where(labels[:,i]==1)
        plt.scatter(Y[idx, 0], Y[idx, 1], c=colors[i], label=classnames[i], marker='.')
    if legend:
        plt.legend()
    if save:
        plt.savefig('fig/train_distri.png')
    if show:
        plt.show()    

def explore(data, A_mean, A_logsigm, topic_word, topic_var, n_topic, n_hidden, name, save=False):
    test_set, test_labels, pred_labels, z, class_names = data
    
    # latent
    A_sigm = np.exp(A_logsigm)
    samples = []
    for i in range(n_topic):
        sample = np.array(list(map(np.random.normal, A_mean[i], A_sigm[i], [50]*n_hidden)))
        samples += [sample.T]
    ss = np.concatenate(samples)
    S = tsne.tsne(ss, 2, ss.shape[0])
    # reconstructed
    theta = np.dot(ss, topic_var[0])+topic_var[1]
    e_t = np.exp(theta - np.max(theta))
    theta = e_t / e_t.sum()
    logits = np.log(np.dot(theta, topic_word))
    SS = tsne.tsne(logits, 2, ss.shape[0])
    # topic divergence (multinomial)
    KL = np.zeros((n_topic, n_topic))
    for i in range(n_topic):
        for j in range(n_topic):
            KL[i,j] = compute_kl_multi(topic_word[i], topic_word[j])
    # plot
    colors = ['orange'] + ['blue'] + ['y'] + ['m'] + ['r']
    plt.close()
    plt.figure(figsize=(15,3))
    gs = gridspec.GridSpec(1, 5)
    gs.update(wspace=0.1, hspace=0.1)

    plt.subplot(gs[0])
    for i in range(n_topic):
        plt.scatter(S[i*50:(i+1)*50, 0], S[i*50:(i+1)*50, 1], c=colors[i], marker='.', 
                    label='%.5f'%(np.mean(pred_labels[:,i] == test_set[:,i])))
    plt.title('Latent prior distributions')
    plt.legend(loc='right')

    plt.subplot(gs[1])
    for i in range(n_topic):
        plt.scatter(SS[i*50:(i+1)*50, 0], SS[i*50:(i+1)*50, 1], c=colors[i], marker='.')
    plt.title('Reconstructed distributions')

    plt.subplot(gs[2])
    plt.imshow(KL, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('KL')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    classes = list(class_names)
    plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = KL.max() / 2.
    for i in range(n_topic):
        for j in range(n_topic):
            plt.text(j, i, format(KL[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if KL[i, j] > thresh else "black")
    
    plt.subplot(gs[3])
    plt.title('Test distribution')
    data_visualize_multilabel(test_set, test_labels, n_topic, class_names, save=False, show=False, legend=False)        

    plt.subplot(gs[4])
    plt.title('Encoded Distribution')
    data_visualize_multilabel(z, test_labels, n_topic, class_names, save=False, show=False, legend=False)        
            
    if save:
        plt.savefig('fig/distribution_%d.png'%name)
    plt.show()

    
    
# kl divergence of multinomial distribution    
def compute_kl_multi(par1, par2):
    kl = 0.5 * (np.sum(par1*(np.log(par1)-np.log(par2))))
    return kl

def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix"""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.sigmoid(inputs)

        class_mask = torch.zeros(N, C).cuda()
        class_mask.scatter_(1, targets.data.long(), 1)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # alpha = self.alpha[targets.data.long()]

        probs = P #(P*class_mask).sum(1).view(-1,1)

        log_p = targets*((probs+1e-12).log())
        np = 1-probs
        log_np = (1-targets)*((np+1e-12).log())
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = (-self.alpha*(torch.pow((np), self.gamma)*log_p+torch.pow((probs), self.gamma)*log_np)).sum(1)
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# if __name__ == '__main__':
    