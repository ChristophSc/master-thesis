# # umls
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=DistMult --task.dir=umls
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=ComplEx --task.dir=umls
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransE --task.dir=umls
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransD --task.dir=umls

# kinship
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g:gpus=1:rtx2080=t python pretrain.py --pretrain_config=DistMult --task.dir=kinship
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g:gpus=1:rtx2080=t python pretrain.py --pretrain_config=ComplEx --task.dir=kinship
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g:gpus=1:rtx2080=t python pretrain.py --pretrain_config=TransE --task.dir=kinship
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g:gpus=1:rtx2080=t python pretrain.py --pretrain_config=TransD --task.dir=kinship

# # wn18rr
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=DistMult --task.dir=wn18rr
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=ComplEx --task.dir=wn18rr
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransE --task.dir=wn18rr
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransD --task.dir=wn18rr

# # wn18
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=DistMult --task.dir=wn18
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=ComplEx --task.dir=wn18
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransE --task.dir=wn18
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransD --task.dir=wn18

# # fb15k237
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=DistMult --task.dir=fb15k237
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=ComplEx --task.dir=fb15k237
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransE --task.dir=fb15k237
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python pretrain.py --pretrain_config=TransD --task.dir=fb15k237