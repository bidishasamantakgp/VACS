# A Deep Generative Model for Code-Switched Text

## Required packages
- `tensorflow 1.13.*`

## Getting Started

### Directory Structure

The following directories provides scripts for VACS 

- `utils`   Contains the files with all the required functions for the model
- `hvae.py` Code for the encoder and decoder architechture of the model
- `vae_lstm-lstm.py`  Run this code to train the model
- `simple_sampling.py`  Run this code to generate sampled sentences after loading a trained model
- `validation.py` Run this code to get validation ppl for the data in /DATA/test_Data/
- `utils/parameters.py` Provides list of parameters to train your model

### Downloads

We used aligned fasttext word vectors of Hindi and English as our embeddings. They can be downloaded directly from [Fasttext website](https://fasttext.cc/docs/en/aligned-vectors.html). 
Alternatively, the following curl commands can be used :
```
curl -o wiki.en.align.vec  https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec
curl -o wiki.hi.align.vec https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hi.align.vec
```

### Dataset

The dataset folder should be located in `DATA` directory. The name of this folder should be specified as `inp_folder_name` in the `utils/parameters.py` The code requires 2 files in the dataset folder:
- `data.txt` Contains the sentences on which the model can be trained on
- `labels.txt` Contains the labels to the sentences in data.txt (We used 0 for english, 1 for hindi, 2 for numbers/symbols/other words)[

[Click here](https://www.dropbox.com/s/3tkjobo8h2zupn8/Dataset_VACS.zip?dl=0) to download the dataset we used.


## Training the generative model

### Prameters
- `name` in `utils/parameters.py` is the name for the experiment. The forlders for training embeddings, model checkpoints and logs will be created using this name
- `num_epochs` Specify the number of epochs you want to train
- Other parameters like layers and hidden state size in encoder and decoder LSTM can be modified.

### Command
`python vae_lstm-lstm.py`


### KL Annealing

### Expected Output


## Sampling sentences

### Parameters
Specify the `number_of_samples` , `sample_sentences_file_name` and `sample_labels_file_name` to specify the number of sentences to be sampled, the path to the sampled sentences and the labels of these sentences respectively.


### Restoring Model

We have to specify the model to be restored in the `simple_sampling.py` file. This file has the following code snipped just after the tensorflow session is initialized:
```
    path="./models_ckpts_"+params.name+"/vae_lstm_model-11900"
    saver.restore(sess,path )
```

The `path` variable has to specify the path to the model checkpoint from where we want to sample the sentences.



