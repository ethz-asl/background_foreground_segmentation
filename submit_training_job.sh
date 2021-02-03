# Example script to submit training job on Leonhard.
# Replace config file after `with` with desired one.
bsub -W 4:00 -n 2 -R "rusage[mem=60000,ngpus_excl_p=1,scratch=10000]" python src/train_binary_segmodel_base.py with experiment_cfg/unet_nyu.yml