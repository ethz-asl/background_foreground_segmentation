# Replace config file after `with` with desired one.
bsub -W 4:00 -n 2 -R "rusage[mem=60000,ngpus_excl_p=1,scratch=10000]" python src/train_binary_segmodel_distillation.py with experiment_cfg/distillation/fast_scnn_from_nyu_to_cla_feature_distillation.yml
