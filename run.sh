#!/bin/bash

#training loop
python3 train.py --model=unet --problem=firstbreak --noise_type=0 --noise_scale=0.25 --device=0 --epochs=100 --lr=5e-5 --batch_size=16


#evaluation to generate the robustness matrix
python3 evaluate_full_matrix.py --model_type=unet --problem=firstbreak --noise_type1=0 --noise_scale1=0.25 --metadata=../metadata/ --savepath=../metadata/evaluation/


#convert tensor board logs to csv 
python3 logs_to_csv.py --problem=firstbreak 

#For hyperparamter search (TPE )
#LR
python3 TPE.py --model=unet --problem=firstbreak --epochs=12 --n_trials=40 
#LR+BS
python3 TPE.py --model=unet --problem=firstbreak --epochs=12 --n_trials=40 --tune_batch_size