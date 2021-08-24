export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH;
export TMPDIR=/media/scratch2/fmilano/logs;
export PYT=/home/fmilano/.virtualenvs/bfseg/bin/python
for id in {1105..1137}; do
	$PYT log_results_from_sacred.py --id $id --save_folder /media/scratch1/fmilano/saved_models_incense/ --evaluate --save_output --save_results;
    #$PYT log_results_from_sacred.py --id $id --save_folder /media/scratch1/fmilano/saved_models_incense/ --model_to_save 100;
    echo $id;
done
