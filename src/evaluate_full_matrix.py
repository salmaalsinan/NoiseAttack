#generating the large matrix with equal spacing 0.05-2.05 every 2
import os
import glob
import torch
from metrics import ConfusionMatrix, RMSE
import torch
from models.build import build_model, model_load_weights
from noiseadding import build_noise_transforms, CombinedTransforms
from data import get_train_val_dataset, get_dataset, get_train_val_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import numpy as np
import argparse
import yaml



def evaluate(model, loader, metrics,problem):
    metrics.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i, (sample) in enumerate(loader):
        x, y = sample['input'].float().to(device) , sample['target'].numpy()
        with torch.no_grad():
            y_pred = model(x)
            if problem == 'firstbreak':
                y_pred = torch.argmax(y_pred, dim=1) # get the most likely prediction
        metrics.add_batch(y, y_pred.detach().cpu().numpy())
        print('_', end='')
    return metrics.get()


def evaluate_robustness(model,problem, metrics,batch_size,workers):
    robustness = np.ones([21,20]) * -1
    for i, noise_type in enumerate([-1,13,14,7,8,9,10,11,12,16,17,18,19,15,0,1,2,3,6,4,5]):
        for j, noise_scale in enumerate([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2.0]):
            if i==0 and j!=0:
                continue
            noise_transforms = build_noise_transforms(noise_type=noise_type, scale=noise_scale)
            denoise_dataset = get_dataset(problem, noise_transforms=noise_transforms)
            _, val_dataset = get_train_val_dataset(denoise_dataset)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
            robustness[i, j] = evaluate(model, valid_loader, metrics,problem)
            print(noise_type, noise_scale, robustness[i, j])    
    return robustness


def RUN(model_type,problem,attack,metadata,savepath,pretrained,noise_type1,noise_scale1,batch_size):
    workers=10

    if problem == 'firstbreak':
        metrics = ConfusionMatrix(2, ["empty", "firstbreak"])
    else:
        metrics = RMSE()
    #print(metrics)

    for noise_type in noise_type1:
        for noise_scale in noise_scale1:
            model = build_model(model_type, problem)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_False_attack_{attack}_pretrained_{pretrained}'
            save_path = os.path.join(metadata, weight_file + '.pkl')
            model.load_state_dict(torch.load(save_path,map_location='cuda:0'))
            model.eval()
            print(weight_file)
            robustness = evaluate_robustness(model, problem, metrics,batch_size,workers)
            np.save(os.path.join(savepath,'full_robustness_' + weight_file + '.npy'),np.array(robustness))
        
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--noise_type1", type=list_of_ints, default=[0,1,2,3,4,5,6])
    parser.add_argument("--noise_scale1", type=list_of_floats, default=[0.05,0.25,0.5,1.0])
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--savepath", type=str, required=True)
    args = parser.parse_args()
  
RUN(args.model_type, args.problem,args.attack,args.metadata,args.savepath, args.pretrained,args.noise_type1, args.noise_scale1, args.batch_size)
