cd ".."
python "pretrain.py" --config=config/config_umls.yaml --pretrain_config=DistMult --log.prefix=umls_distmult_
python "pretrain.py" --config=config/config_umls.yaml --pretrain_config=TransE --log.prefix=umls_transe_
python "gan_train.py" --config=config/config_umls.yaml --g_config=DistMult --d_config=TransE --log.prefix=umls_adv_
pause