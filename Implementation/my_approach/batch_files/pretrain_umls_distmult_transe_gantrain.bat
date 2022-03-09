cd ".."
python "pretrain.py" --config=config/config_umls.yaml --pretrain_config=DistMult --log.prefix=umls_distmult_
python "pretrain.py" --config=config/config_umls.yaml --pretrain_config=TransE --log.prefix=umls_transe_
python "gan_train.py" --config=config/config_umls.yaml --g_config=DistMult --d_config=TransE --log.prefix=umls_
pause


########## MIT Info ##########
ccsalloc -s now --duration 2h --res=rset=1:ncpus=8:mem=20g:gpus=1:rtx2080=t python gan_train.py --task.dir=umls --g_config=DistMult --d_config=TransE 
ccsalloc -s now --duration 2h --res=rset=1:ncpus=8:mem=20g:gpus=1:=gtx1080=t python gan_train.py --task.dir=umls --g_config=DistMult --d_config=TransE 
ccsalloc -s now --duration 2h --res=rset=1:ncpus=8:mem=20g:gpus=1:=tesla=t python gan_train.py --task.dir=umls --g_config=DistMult --d_config=TransE 



########## OHNE Info ##########
ccsalloc -s now --duration 24h --res=rset=1:ncpus=8:mem=20g:gpus=1:gtx1080=t python gan_train.py --g_config=DistMult --d_config=TransE
