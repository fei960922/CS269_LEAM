# CS269_LEAM

Project for CS269_2019Spring

## How to run

see main.ipynb

## Files

- data_process_reuters.ipynb : download and process reuters dataset.
- main.ipynb : main file to run
- src/
  - model.py : 
    - Class Embedding : implement word embedding (pre-defined) along with label-embedding attention --- can change to bag of word or other predefined word embedding
    - Class Classifier : MLP + Relu classifier --- can change to LSTM / CNN / DNN
    - Class Leam_Classifier : combine two above
  - train.py : Code for training and testing 
  - main.py : Main file to run in terminal
  - util.py : Util functions. --- please add all other function in util
  - zsdm.py : Please omit
- data/ : will be produced by data_process_*.ipynb, please store every later data in this folder

## Reference 

- LEAM : See reference folder
- [Github LEAM](https://github.com/guoyinwang/)
- [Chinese blog about LEAM](https://zhuanlan.zhihu.com/p/54734708)