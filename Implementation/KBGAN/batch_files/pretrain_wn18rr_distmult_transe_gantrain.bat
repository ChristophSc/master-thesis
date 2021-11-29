cd ".."
python "pretrain.py" --config=config_wn18rr.yaml --pretrain_config=DistMult --log.prefix=wn18rr_distmult_
python "gan_train.py" --config=config_wn18rr.yaml --g_config=DistMult --d_config=TransE --log.prefix=wn18rr_
pause