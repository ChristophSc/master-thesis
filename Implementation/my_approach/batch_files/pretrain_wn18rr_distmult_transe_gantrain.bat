cd ".."
python "pretrain.py" --config=config/config_wn18rr.yaml --pretrain_config=DistMult
python "pretrain.py" --config=config/config_wn18rr.yaml --pretrain_config=TransE
python "gan_train.py" --config=config/config_wn18rr.yaml --g_config=DistMult --d_config=TransE
pause