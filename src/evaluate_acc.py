#generating the  matrix to get the accuracy metric of models and tasks
import os
import pandas as pd
import torch
from metrics import ConfusionMatrix, RMSE
import torch
from models.build import build_model
from noiseadding import build_noise_transforms
from data import get_train_val_dataset, get_dataset, get_train_val_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm

#get metrics table

def evaluate_model(problem,model, loader, metrics):
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


# Main experiment loop
def run_metrics(problem, model_type, batch_size,attack,pretrained):
    clip=False
    output_path= f'../metadata/metric_tables/'
    if attack=='none':
        metadata=f'../metadata/single_source/50E/NoAttack/'
        #metadata1=f'../metadata/50E/NoAttack/'
        metadata1=f'../metadata/metadata/{model_type}/NoAttack/'

    else:
        metadata=f'../metadata/single_source/Attack_{attack}/'
        metadata1=f'../metadata/metadata/{model_type}/Attack_{attack}/'

    if pretrained == True:
            metadata=os.path.join(metadata, 'pretrained')
            metadata1=os.path.join(metadata1, 'pretrained')

    if problem == 'firstbreak':
        metrics = ConfusionMatrix(2, ["empty", "firstbreak"])
    else:
        metrics = RMSE()
    print(metrics)
    workers=10
    results = []
    noise_types = list(range(20))  # 0 to 19
    noise_scales = [0.05,0.1,0.15,0.2,0.25,0.5,1.0,2.0]
    
    output_csv = f'{model_type}_{problem}_attack_{attack}_pretrained_{pretrained}_metrics_table.csv'

    for noise_type in tqdm(noise_types, desc='Noise Types'):
        row = {'noise_type': noise_type}
        for noise_scale in noise_scales:
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(model_type, problem)
            model.to(device)
            weight_path = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_{clip}_attack_{attack}_pretrained_{pretrained}'
            
            try:
                save_path = os.path.join(metadata, weight_path + '.pkl')
                model.load_state_dict(torch.load(save_path,map_location='cuda:0'))

            except FileNotFoundError:
                
                try:
                    save_path = os.path.join(metadata1, weight_path + '.pkl')

                    model.load_state_dict(torch.load(save_path,map_location='cuda:0'))
                
                except FileNotFoundError:
                    row[f"{noise_scale}"] = None
                    continue
        
            model.eval()
            #print('model loaded')

            # Generate noisy data
            
            noise_transforms = build_noise_transforms(noise_type=noise_type, scale=noise_scale)
            denoise_dataset = get_dataset(problem, noise_transforms=noise_transforms)
            _, val_dataset = get_train_val_dataset(denoise_dataset)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                      num_workers=workers)
            
            #print('data loaded')

            # Evaluate
            acc = evaluate_model(problem,model, valid_loader,metrics)
            print(noise_type, noise_scale, acc)
            #print(np.shape(acc))
            row[f"{noise_scale}"] = acc
        #return acc
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_path, output_csv), index=False)
    print(f"Saved results to {output_csv}")
    #return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=False)
 

    args = parser.parse_args()


run_metrics(args.problem, args.model_type, args.batch_size,args.attack,args.pretrained)