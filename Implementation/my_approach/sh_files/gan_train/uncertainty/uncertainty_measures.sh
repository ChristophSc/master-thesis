# # entropy
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=entropy
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=entropy
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=entropy
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=entropy

# # least_confidence
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=least_confidence
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=least_confidence
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=least_confidence
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=least_confidence

# # confidence_margin
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_margin
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_margin
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_margin
# ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_margin

# confidence_ratio
ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransE --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_ratio
ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=DistMult --d_config=TransD --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_ratio
ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransE  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_ratio
ccsalloc -s now --duration 3h --res=rset=1:ncpus=8:mem=20g python gan_train.py --task.dir=kinship --g_config=ComplEx --d_config=TransD  --adv.sample_type=uncertainty_softmax --adv.use_pretrained_models=false --adv.measure_type=confidence_ratio

