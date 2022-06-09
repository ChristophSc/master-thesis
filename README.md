# USGAN 
- Description: Incorporation of uncertainty information in an existing negative sampling approach 
- Master Thesis Topic: Sampling of Negative Triples for Knowledge Graph Embeddings by Uncertainty
- Name: Christoph Schäfer

In USGAN, uncertainty is used in the process of negative sampling.  
It replaces the original sampling method of the generative adversarial network-based approach [KBGAN](https://arxiv.org/abs/1711.04071).


## Installation

```
git clone https://github.com/ChristophSc/master-thesis.git
python --version
Python 3.6.8
pip install -r requirements.txt
```

## Configuration
All settings for pre-training and adversarial training can be made in the file `config/config.yaml`.  
The listed parameters can be edited before starting the training or overwritten with command line parameters.

## Execution

```
# start pre-training
python pretrain.py 

# start adversarial training
python gan_train.py
```

## Creation of Evaluation Graphs
If logging is enabled via the `config.yaml` file, graphs are created and results are logged during training in the directory `logs/`.


If combined graphs of the results and the different model pairs are to be created, the logs can be placed in the `logs/backup/` directory.
By executing the following command, the combined graphs will then be placed under `combined_figures/`.
```
python CombinedGraphCreator.py
```
