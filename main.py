import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from src.model import *
from src.utils import *
from src.trainer import *

def write(cache, name, max_len=80, imp=1.5e-2):
    f.write('<p style="background-color:green;">%s</p>'%name)
    for pred, true, sent, recon in cache:
        if (pred[true==1] != 1).all():
            f.write('<p style="background-color:red;">All Miss</p>')
        elif (pred != true).any():
            f.write('<p style="background-color:blue;">Partial Wrong</p>')                    
        f.write('<p>prediction: {}, true: {}</p>'.format(class_names[pred==1], class_names[true==1]))
        f.write('<p>')

        f.write('<p>')
        for i, word in enumerate(sent):
            if i >= max_len:
                break
            if beta[i] > imp:
                f.write('<mark class="red">{}</mark> '.format(vocab[word]))
            else:
                f.write('{} '.format(vocab[word]))                
        f.write('</p>')    
        f.write('<HR SIZE=5>')

def main():
    data_dir = 'data/reuters'
    vocab_url = os.path.join(data_dir, 'vocab.pkl')
    train_url = os.path.join(data_dir, 'train')
    test_url = os.path.join(data_dir, 'test')
    vocab = pickle.load(open(vocab_url, 'rb'))
    vocab = list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0]
    vocab_size = len(vocab)

    train_set, train_labels, class_names = dataset(train_url)
    test_set, test_labels, _ = dataset(test_url)

    model = Leam_Classifier(vocab_size, len(class_names), 256, 256, 10, 
                            n_layer=1, dropout_rate=0.8, embpath=None, label_att=True, multilabel=True)            
    trainer = Trainer(batch_size=128, num_epoches=100, learning_rate=1e-3, valid_freq=10, model_type='embed')

    trainer.set_vocab(vocab)
    trainer.set_validation(test_set, test_labels)
    trainer.init_model(model)
    train_prob, train_beta = trainer.fit(train_set, train_labels, class_names)
    
    # validation (test)
    val_prob, val_beta = trainer.predict(test_set)
    val_pred = val_prob > 0.5
    # test not seen in training (task)
    task_url = os.path.join(data_dir, 'task')
    seq_task, seq_labels, _ = dataset(task_url, monitor=False)
    test_prob, test_beta = trainer.predict(seq_task)
    test_pred = test_prob > 0.5
    multilabel_eval(seq_labels, test_pred, full=True)
    
    # shwo results and highlights
    trains = zip(train_prob>0.5, train_labels, train_set, train_beta)
    vals = zip(val_pred, test_labels, test_set, val_beta)
    tests = zip(test_pred, seq_labels, seq_task, test_beta)

    with open('res.html', 'w', encoding='gbk') as f:
        write(tests, 'Test', max_len=80, imp=1.5e-2)
        write(vals, 'Validation', max_len=80, imp=1.5e-2)
        write(trains, 'Train', max_len=80, imp=1.5e-2)

if __name__ == "__main__":
    main()