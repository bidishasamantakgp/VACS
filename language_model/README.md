# tf-lstm-char-cnn

The language model code for extrinsic evaluation of generated Code-Switched (gCS) sentences is done with the help of [tf-lstm-char-cnn](https://github.com/mkroutikov/tf-lstm-char-cnn). We train a LM on different combinations of monolingual and generated sentnces (please refer to paper for more details) and test them on real Code-Switched sentences. It is evaluated with `perplexity` scores. Lower perplexity means generated sentences are more similar to real code-switched sentences.
  
## Requirements
- tensorflow version 1.0 

## Running

```
python train.py -h
python evaluate.py -h
```

## Data

Data should be present in the `data` folder. It should contain 3 files :
- `train.txt`
- `valid.txt`
- `test.txt`

## Training

```
python train.py 
```

Different hyperparameters can be set by passing them as command line arguments. Please refer to `train.py` for more details.

## Evaluate

```
python evaluate.py --load_model cv/epoch024_4.4962.model
```
evaluates this model on the test dataset



## Pre-trained model

We are publishing pre-trained model files to the following two Language Model experiments we did with generated sentences from VACS model. :
- Mono | VACS
- VACS | Mono

The following ZIP folder contains pre-trained model and the the dataset used for the LM experiment. [Pre-trained model](https://www.dropbox.com/s/pid65mgu76vxxn1/LM_Experiments_model_data.zip?dl=0)

This model was trained with the default parameters and acheved the accuracy of the published result.


