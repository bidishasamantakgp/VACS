# A Deep Generative Model for Code-Switched Text

## Required packages
- `tensorflow 1.13.*`

## Quick start

The following directories provides scripts for VACS 

- `utils`   Contains the files with all the required functions for the model
- `hvae.py` Code for the encoder and decoder architechture of the model
- `vae_lstm-lstm.py`  Run this code to train the model
- `simple_sampling.py`  Run this code to generate sampled sentences after loading a trained model
- `validation.py` Run this code to get validation ppl for the data in /data/test_Data/
- `utils/parameters.py` Provides list of parameters to train your model

## Command to train the generative model

`python vae_lstm-lstm.py`
