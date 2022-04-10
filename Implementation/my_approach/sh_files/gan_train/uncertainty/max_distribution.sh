################### not_pretrained
# umls
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=umls --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=umls --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=umls --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
# ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=umls --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false

# wn18rr
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18rr --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18rr --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18rr --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18rr --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false

# # wn18
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18 --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18 --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18 --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=wn18 --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false

# fb15k237
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=fb15k237 --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=fb15k237 --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=fb15k237 --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
ccsalloc -s now --duration 1d --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=fb15k237 --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false
