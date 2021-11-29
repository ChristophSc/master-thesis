# Master Thesis

### Topic: Sampling of Negative Triples for Knowledge Graph Embeddings by Uncertainty

## Structure

- `./research/...`: all PDF files and with comments and edits

- `./writing/proposal`: contains all necessary LaTeX files for the proposal
- `./writing/master-thesis`: contains all necessary LaTeX files for the master thesis

- `./implementation/...`: contains all source code of my approach and original approaches
- `./implementation/my_approach`: contains source code of my implementation
- `./implementation/original_models`: original source code of some other approaches like `KBGAN`, `NSCaching`, ...
- `./implementation/KBGAN`: adjusted KBGAN that works with my Python and Pytorch version (no cuda, adjustments to newer PyTorch and Python version)

- `./results.xlsx`: Excel file with all documented results from my approach, including MRR, Hit10 and other statistics

## Structure of the implementation

- `./implementation/my_approach/pretrain.py`: pretrains the model
- `./implementation/my_approach/gan_train.py`: adversarial training which requires pretrained models for generator and discriminator
- `./implementation/my_approach/batch_files`: contains batch files to run pretraining/adverarial training with different parameters
- `./implementation/my_approach/data`: contains datasets. Train, valid and test tests are in separated files
- `./implementation/my_approach/config`: contains configuration files for each dataset, default one is `config.yaml`
- `./implementation/my_approach/models`: output folder for pretrained models
- `./implementation/my_approach/logs`: log files for pretraining and adverarial training processes
- `./implementation/my_approach/graphs`: evaluation figures for discriminator losses, rewards, MRR and Hit@10 over time for each training
- `./implementation/my_approach/plant_uml`: PlantUML and images of different UML diagrams

## Usage

- Pretrain: python3 pretrain.py --config=config_<dataset_name>.yaml --pretrain_config=<model_name> (this will generate a pretrained model file)
- Adversarial train: python3 gan_train.py --config=config_<dataset_name>.yaml --g_config=<G_model_name> --d_config=<D_model_name> (make sure that G model and D model are both pretrained)

