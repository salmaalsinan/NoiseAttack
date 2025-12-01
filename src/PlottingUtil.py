
import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
from pathlib import Path
from data import get_train_val_dataset, get_dataset
from noiseadding import build_noise_transforms
from models.build import build_model

import matplotlib.colors as colors
from skimage import transform
from matplotlib import patches
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

import matplotlib.patches as mpatches

def Plot_model(model_type,problem,noise_type, noise_scale,noise_type2, noise_scale2,attack='none',
               pretrained=False, save_path=None,folder=None,savefigure=False,prefix='',loader='Validation'): 
    
    if save_path is None:
        save_path = Path("../images/")
        save_path.mkdir(parents=True, exist_ok=True)

    x,y= Sample_noise(problem,noise_type2, noise_scale2,loader)
    
  
    model = build_model(model_type, problem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_True_attack_{attack}_pretrained_{pretrained}'
        _path = os.path.join(folder, weight_file + '.pkl')
        model.load_state_dict(torch.load(_path,map_location='cuda:0'))

    except FileNotFoundError:
        weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_False_attack_{attack}_pretrained_{pretrained}'
        _path = os.path.join(folder, weight_file + '.pkl')
        model.load_state_dict(torch.load(_path,map_location='cuda:0'))

    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        if problem == 'firstbreak':
            y_pred = torch.argmax(y_pred, dim=1) # get the most likely prediction
    
    plt.figure(figsize=(18,9))
    plt.imshow(torchvision.utils.make_grid(x.cpu(), padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    if savefigure ==True:
        plt.savefig(f'{save_path}/{prefix}input_gen_N{noise_type2}_S{noise_scale2}_{problem}.jpg',dpi=1000, bbox_inches='tight',transparent=True)

    if problem == 'firstbreak':
        plt.figure(figsize=(18,9))
        plt.imshow(torchvision.utils.make_grid(y.float().unsqueeze(1), padding=0)[0][None,...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
        plt.title('Target') 
        plt.xticks([])
        plt.yticks([])
        if savefigure ==True:
            plt.savefig(f'{save_path}/{prefix}target_gen_N{noise_type2}_S{noise_scale2}_{problem}.jpg',dpi=1000, bbox_inches='tight',transparent=True)

        
        plt.figure(figsize=(18,9))
        plt.imshow(torchvision.utils.make_grid(y_pred.detach().cpu().float().unsqueeze(1), padding=0)[0][None,...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
        plt.title('Predicted')
        plt.xticks([])
        plt.yticks([])
        if savefigure ==True:
            plt.savefig(f'{save_path}/{prefix}predicted_gen_N{noise_type2}_S{noise_scale2}_{problem}.jpg',dpi=1000, bbox_inches='tight',transparent=True)

    else:
        plt.figure(figsize=(18,9))
        plt.imshow(torchvision.utils.make_grid(y.float(), padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
        plt.title('Target')
        plt.xticks([])
        plt.yticks([])
        if savefigure ==True:
            plt.savefig(f'{save_path}/{prefix}target_gen_N{noise_type2}_S{noise_scale2}_{problem}.jpg',dpi=1000, bbox_inches='tight',transparent=True)

    
        plt.figure(figsize=(18,9))
        plt.imshow(torchvision.utils.make_grid(y_pred.float(), padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
        plt.title('Predicted')
        plt.xticks([])
        plt.yticks([])
        if savefigure ==True:
            plt.savefig(f'{save_path}/{prefix}predicted_gen_N{noise_type2}_S{noise_scale2}_{problem}.jpg',dpi=1000, bbox_inches='tight',transparent=True)

    
        plt.figure(figsize=(18,9))
        plt.imshow(torchvision.utils.make_grid(y_pred.float()-sample['target'].float(), padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
        plt.title('Error/Difference')
        plt.xticks([])
        plt.yticks([])
        if savefigure ==True:
            plt.savefig(f'{save_path}/{prefix}diff_gen_N{noise_type2}_S{noise_scale2}_{problem}.jpg',dpi=1000, bbox_inches='tight',transparent=True)


    plt.show()




def Sample_noise(problem,noise_type2, noise_scale2,loader='Validation'): 

    batch_size=8
    workers=4
    noise_transforms = build_noise_transforms(noise_type2, noise_scale2)
    print(f'noise_type2={noise_type2}  noise_scale2={noise_scale2}')
    denoise_dataset = get_dataset(problem, noise_transforms=noise_transforms)
    train_dataset, val_dataset = get_train_val_dataset(denoise_dataset)
    if loader != 'Validation':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    else:
        train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    sample = next(iter(train_loader))
    x, y = sample['input'].float().cuda(), sample['target']

    return x,y

def Plot_predictions(x, y, model_type,problem,noise_type, noise_scale,attack='none',pretrained=False,
                     folder=None,ax=None,save_path=None,savefigure=False,prefix=''): 
    y=y.numpy()
    model = build_model(model_type, problem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_True_attack_{attack}_pretrained_{pretrained}'
        path = os.path.join(folder, weight_file + '.pkl')
        model.load_state_dict(torch.load(path,map_location='cuda:0'))
    
    except FileNotFoundError:
        weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_False_attack_{attack}_pretrained_{pretrained}'
        path = os.path.join(folder, weight_file + '.pkl')
        model.load_state_dict(torch.load(path,map_location='cuda:0'))
    
    model.eval()

    
    with torch.no_grad():
        y_pred = model(x)
        if problem == 'firstbreak':
            y_pred = torch.argmax(y_pred, dim=1) # get the most likely prediction
    
    if ax is None:
        fig= plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)

    if problem == 'firstbreak':
        im=ax.imshow(torchvision.utils.make_grid(y_pred.detach().cpu().float().unsqueeze(1), padding=0)[0][None,...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    else:

        M=np.abs(y_pred.detach().cpu().float()-y) #plot absolute difference

        im=ax.imshow(torchvision.utils.make_grid(M, padding=0)[0][None, ...].permute((1, 2, 0)), cmap='hot', vmin=0, vmax=0.8)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        

    if savefigure == True:
        if save_path is None:
            save_path = Path("../images/")
            save_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{save_path}/Prediction_{model_type}_{problem}_noisescale_{noise_scale}_noisetype_{noise_type}_pretrained_{pretrained}_{prefix}.jpg',dpi=1000, bbox_inches='tight',transparent=True)
    
    return im
    


def Plot_predictions_grid(noise_scale, problem, noise_type2,noise_scale2, idx=None,folder=None, save_path=None, 
                          attack='none', pretrained=False, savefigure=False,loader='Validation'): 
    


    if idx is None: 
        idx={-1:'Clean',7:'Gaussian',8:'Colored',9:'Linear',10:'Random (fft)',11:'Hyperbolic',12:'Bandpassed',
            130: 'Lowpass Random', 14:'Trace-wise',13: 'Lowpass',
                    0:'Gaussian + Color', 
                    1:'X: Gaussian + Colored + Linear',
                    2:'X + Gaussian(f-k)',
                    3:'Random + Structured',
                    6:'X + Trace-wise',
                    4:'X + Lowpass', 
                    5:'X + Lowpass + Trace-wise', 
                    15:'Gaussian + Color + Gaussain (f-k)', 
                    16:'Gaussian (x-t)+(f-k)', 
                    17:'Structured', 
                    18: 'Colored + Linear',
                    19: 'Gaussian + Linear'}

    x,y= Sample_noise(problem,noise_type2, noise_scale2,loader)

    if folder is None:  
        folder =f'../metadata'

    
    fig= plt.figure(figsize=(20,11))#, constrained_layout=True)

    gs = gridspec.GridSpec(9, 3, height_ratios=[1, 1, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], hspace=0.5, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, :])  # row 0, all columns
    ax2 = fig.add_subplot(gs[1, :])  # row 1, all columns

    axes = []
    for row in range(3, 9):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

    ax1.imshow(torchvision.utils.make_grid(x.detach().cpu(), padding=0)[0][None, ...].permute((1, 2, 0)), 
            cmap='seismic', vmin=-1, vmax=1)


    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-100,150, 'Input', fontsize=18, fontweight='bold', horizontalalignment='right')
    t=ax1.text(1850,100,'Noise type =', fontsize=18, weight='bold')
    if noise_type2 in [1,2,3,4,5,6]:
        ax1.annotate(f'{idx[noise_type2]}',xycoords=t,xy=(1,-1.1), fontsize=18, weight='bold',color='r',
                    verticalalignment='bottom')
    else:
        ax1.annotate(f'{idx[noise_type2]}',xycoords=t,xy=(1,0), fontsize=18, weight='bold',color='r',
                    verticalalignment='bottom')

    ax1.annotate('\u03B2 = '+f'{noise_scale2}',xycoords=t,xy=(0,-1), fontsize=18, weight='bold',
                verticalalignment='bottom')

    if problem == 'firstbreak':

        ax2.imshow(torchvision.utils.make_grid(y.unsqueeze(1), padding=0)[0][None,...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
    else:

        ax2.imshow(torchvision.utils.make_grid(y.float(), padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic', vmin=-1, vmax=1)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.text(-100,150, 'Target', fontsize=18, fontweight='bold', horizontalalignment='right')

    i=0
 
    for n in [7,10,8,12,9,11]:
        
        Plot_predictions(x, y, model_type='unet',problem=problem,noise_type=n, noise_scale=noise_scale,attack=attack,
                        pretrained=pretrained,folder=folder,ax=axes[i])
        i+=1

        Plot_predictions(x, y, model_type='swin',problem=problem,noise_type=n, noise_scale=noise_scale,attack=attack,
                        pretrained=pretrained,folder=folder,ax=axes[i])
        i+=1
        im=Plot_predictions(x, y, model_type='restormer',problem=problem,noise_type=n, noise_scale=noise_scale,attack=attack,
                        pretrained=pretrained,folder=folder,ax=axes[i])
        i+=1


    j=0
    t1=axes[0].text(-100,150, f'{idx[7]}', fontsize=18, horizontalalignment='right')
    j=1
    xx=[7,10,8,12,9,11]
    for i in [3,6,9,12,15]:
        axes[i].text(-100,150, f'{idx[xx[j]]}', fontsize=18, horizontalalignment='right')
        j+=1


    axes[0].annotate('Prediction',xycoords=t1,xy=(-0.2,3), fontsize=18, weight='bold',
                verticalalignment='bottom')
    axes[0].set_title('(i) UNet $_{\u03B2= 1.0}$ ', fontsize=18, weight='bold')
    axes[1].set_title('(ii) Swin-U $_{\u03B2= 1.0}$ ', fontsize=18, weight='bold')
    axes[2].set_title('(iii) Restormer $_{\u03B2= 1.0}$ ', fontsize=18, weight='bold')


    big_ax = fig.add_subplot(gs[3:, :], frameon=False)
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel('Training Noise Type', labelpad=150, fontsize=18, rotation=90, va='center',fontweight='bold')

    if problem=='denoise':
        
        cbar_ax = fig.add_axes([0.7, 0.7, 0.2, 0.02])


        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Absolute Error', size=16, weight='bold',labelpad=-60)

    plt.tight_layout()

    
    if noise_scale <0.2:
        level='low'
    elif noise_scale >0.99:
        level='high'
    else:
        level='medium'

    if savefigure == True:
        if save_path is None:
            save_path = Path(f"../images/{level}/")
            save_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(f'{save_path}/Noises_inference_{problem}_attack_{attack}_{pretrained}_N{noise_type2}_s{noise_scale2}_t_s{noise_scale}.jpg',dpi=1000, bbox_inches='tight',transparent=True)
    plt.show()



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def Plot_Full_Robustness(noise_type,problem, model_type, attack='none',pretrained=False, 
                         scale=[0.05, 0.25],epoch=50,cb=True,save_path=None,savefigure=False,I='kaiser'):
    
    idx= ['Clean', 'Gaussian','Colored','Linear','Gaussian (f-k)','Hyperbolic','Bandpassed',
          'Gaussian + Gaussian (f-k)','Linear + Hyperbolic','Linear + Colored','Gaussian + Colored', 
          'X: Gaussian + Colored + Linear', 'X + Gaussian (f-k)','X + Gaussian (f-k) + Hyperbolic',
         'X + Trace-wise','X + Lowpass','X + Trace-wise + Lowpass']
    

    if noise_type <7:
        T='compound'
    else:
        if noise_type >=15:
            T='Compound'
        else:
            T='Single'

    snr=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    ii={0.05:0,0.1:1,0.2:2,0.3:3,0.4:4,0.5:5,0.6:6,0.7:7,0.8:8,0.9:9,1.0:10,1.1:11,1.2:12,1.3:13,1.4:14,1.5:15,1.6:16,1.7:17,1.8:18,1.9:19,2.0:20}

    if epoch==50:
        metadata=f'../metadata/50E/evaluation/'
    else:
        metadata=f'../metadata/evaluation/'
  
    if noise_type==-1:
        SCALE=[0.0]
    else:
        SCALE=scale
    for noise_scale in SCALE:
        if noise_type==-1:
            noise_scale=0.0
        
        try:
            weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_True_attack_{attack}_pretrained_{pretrained}'
            R=np.load(os.path.join(metadata,'full_robustness_' + weight_file + '.npy'))
        except FileNotFoundError:
            weight_file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_False_attack_{attack}_pretrained_{pretrained}'
            R=np.load(os.path.join(metadata,'full_robustness_' + weight_file + '.npy'))
    
        R=np.delete(R,1,0)
        R=np.delete(R,1,0)

        if T =='Single':
            NN=6
        else:
            NN=9
          
        mapping = {0: 10,1: 11,2: 12,3: 13,4: 15,5: 16,6: 14}

        n = mapping.get(noise_type, noise_type - NN)


        ax=plt.gca()
            

        if problem=='denoise':
            cm = plt.get_cmap('gist_rainbow')
            new_cmap = truncate_colormap(cm, 0, 0.95)
            c='hsv'
            mn=0
            mx=0.30
            R[0][1:]=R[1][1:]
            L='RMSE'
            E='max'
        else:
            cm = plt.get_cmap('hsv_r')
            new_cmap = truncate_colormap(cm, 0.3, 1)
            c=new_cmap
            mn=0.5
            mx=1
            R[0][1:]=R[1][1:]
            L='IoU'
            E='min'
        t = transform.resize(R, (len(idx),44), preserve_range=True,order=1,mode='edge').astype(R.dtype)
        im=plt.imshow(t, cmap=c,vmin=mn, vmax=mx, aspect='auto',interpolation=I,interpolation_stage='rgba')
        plt.yticks(np.arange(len(idx)),labels=idx)
        plt.xticks(np.arange(0,44,4),labels=snr[::2])
        plt.xlabel('Noise Scale',fontsize=14, fontweight='bold')
        plt.ylabel('Noise Type',fontsize=14, fontweight='bold')
        if noise_type==-1:
            idx0=(-0.5,-0.5)
        else:
            if noise_type >=19:
                idx0=(-10,-10)
            else:
                try: 
                    idx0 = (ii[noise_scale+0.05]+1, n-0.5)
                except KeyError:
                    idx0 = (ii[noise_scale]*2-0.5, n-0.5)

                if noise_scale==0.05:
                    idx0 = (-0.5, n-0.5)
        #print(idx0)
        if noise_type==-1:
            Col='k'
        else:
            Col='w'
        ax.add_patch(
            patches.Rectangle(idx0,
                    1.0,
                    1.0,
                    edgecolor=Col,
                    fill=False,
                    lw=3
                ) )
        ax.add_patch(
            patches.Rectangle((0.5,-1),
                    44.0,
                    1.5,
                    edgecolor='w',
                    fill=True,
                    facecolor='w',
                    lw=1
                ) )
        if cb==True:
            plt.colorbar(extend=E).set_label(L,size=12, weight='bold', color='k')

        if savefigure==True:
            if save_path is None:
                save_path = Path(f"../images/Robustness/{T}")
                save_path.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(f'{save_path}/Full_Robustness_Matrix_Interpolated_{I}_{problem}_{model_type}_attack_{attack}_noisetype{noise_type}_noisscale_{int(noise_scale*100)}_pt_{pretrained}.jpg',dpi=1000, bbox_inches='tight',transparent=True)
            
    return im , R


def replace_spikes(df, column):
    # Calculate mean and standard deviation
    mean = df[column].mean()
    std_dev = df[column].std()
    
    # Define thresholds
    upper_threshold = mean + 2 * std_dev
    lower_threshold = mean - 2 * std_dev
    

    # Avoid the first 10 values
    for i in range(10, len(df) - 1):
        # Detect upper spikes
        if (df[column].iloc[i] > upper_threshold and 
            df[column].iloc[i-1] < upper_threshold and 
            df[column].iloc[i+1] < upper_threshold):
            # Replace upper spike with the average of the surrounding values
            df.loc[i, column] = (df[column].iloc[i-1] + df[column].iloc[i+1]) / 2

        # Detect lower spikes
        elif (df[column].iloc[i] < lower_threshold and 
              df[column].iloc[i-1] > lower_threshold and 
              df[column].iloc[i+1] > lower_threshold):
            # Replace lower spike with the average of the surrounding values
            df.loc[i, column] = (df[column].iloc[i-1] + df[column].iloc[i+1]) / 2

    return df


def smooth(x, w):
    if w ==0: 
        y_smooth =x
    else:
        y_smooth=np.convolve(x, np.ones(w), 'valid') / w
    return y_smooth


def read_logs(model_type, problem,noise_type, noise_scale, attack='none', pretrained=False,folder=None,dataclip=False,prefix=''):
    
    if folder==None:
        m=f'../../tensorboard/csv/'
    else:
        m= folder

    METADATA = os.path.join(m, f'{problem}/NoAttack/') 
  
    if noise_type==-1:
        try: 
            file = f'{prefix}{model_type}_{problem}_noisetype_{noise_type}_noisescale_0.0_dataclip_{dataclip}_attack_{attack}_pretrained_{pretrained}.csv'
            f= pd.read_csv(os.path.join(METADATA,  file), index_col=0)
            f= replace_spikes(f, f.columns[1])
            f= replace_spikes(f, f.columns[2])
        except FileNotFoundError:
            print('no file2')
            pass
    else:
        try:
            file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_False_attack_{attack}_pretrained_{pretrained}.csv' 
            f= pd.read_csv(os.path.join(METADATA,  file), index_col=0)
            f= replace_spikes(f, f.columns[1])
            f= replace_spikes(f, f.columns[2])
        except FileNotFoundError:
            file = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_True_attack_{attack}_pretrained_{pretrained}.csv' 
            f= pd.read_csv(os.path.join(METADATA,  file), index_col=0)
            f= replace_spikes(f, f.columns[1])
            f= replace_spikes(f, f.columns[2])

    return f


def Plot_logs_nscale(problem,model_type,noise_type,attack='none',ax=None,c='PuBu_r',ts_factor=0.5,a=0.5,pretrained=False,L='-',folder=None):
    
    if ax is None:
        fig, ax= plt.subplots(nrows=1, ncols=2, sharex= True, sharey= False)
        a=1
    else:
        pass
    cmap = plt.get_cmap(c)
    colors = cmap(np.linspace(0,1,9))
    ax[0].set_prop_cycle(color=colors)
    ax[1].set_prop_cycle(color=colors)
    try:
        log = read_logs(model_type = model_type, noise_type=-1,noise_scale=0.0,problem=problem,
                        attack=attack,pretrained=pretrained,folder=folder,dataclip=False, )

        ax[0].plot((log[log.columns[0]]+1).ewm(alpha=(1 - ts_factor)).mean(),log[log.columns[1]].ewm(alpha=(1 - ts_factor)).mean(),label='clean',
                alpha=a,linestyle=L) 
    
        ax[1].plot((log[log.columns[0]]+1).ewm(alpha=(1 - ts_factor)).mean(),log[log.columns[2]].ewm(alpha=(1 - ts_factor)).mean(),label='clean',
                alpha=a,linestyle=L) 
        m=log.columns[2]
    except UnboundLocalError:
        pass
    for noise_scale in [0.05,0.1,0.15,0.2,0.25,0.5,1.0,2.0]:

        try:
            log = read_logs(model_type = model_type, noise_type=noise_type, noise_scale=noise_scale,problem=problem,attack=attack,pretrained=pretrained,folder=folder)

            ax[0].plot((log[log.columns[0]]+1).ewm(alpha=(1 - ts_factor)).mean(),(log[log.columns[1]]).ewm(alpha=(1 - ts_factor)).mean(),label=noise_scale, alpha=a,linestyle=L) 
            ax[1].plot((log[log.columns[0]]+1).ewm(alpha=(1 - ts_factor)).mean(),(log[log.columns[2]]).ewm(alpha=(1 - ts_factor)).mean(),label=noise_scale, alpha=a,linestyle=L) 
            m=log.columns[2]
        except:
            print('no file')
            continue
    plt.subplots_adjust(right=2)
    
    for i in range(0,2):
        ax[i].set_xlim(0,100)
        if a ==1:
            ax[i].set_title(f'{model_type},{problem}, noise type {noise_type} ', fontsize=14)
            ax[i].set_xlabel('Epoch',fontsize=14,fontweight='bold')
            h,l = ax[0].get_legend_handles_labels()
            L=ax[0].legend([h[0], h[1],h[2], h[3],h[4], h[5],h[6], h[7],h[8]],
                     ['Clean','0.05','0.1','0.15','0.2','0.25','0.5','1.0','2.0'],
                     loc='upper center', bbox_to_anchor=(1.2, -0.15),
                     fontsize=12,
                     labelcolor='k',facecolor='none',edgecolor='none',ncol=9)  

            for b in range (0,9,1):
                L.get_lines()[b].set_linewidth(3)
                L.get_lines()[b].set_alpha(1)

    ax[0].set_ylabel('Loss', fontsize=14,fontweight='bold')
  
    plt.subplots_adjust(right=2)


def Plot_logs_Mean(model_type,attack='none',pretrained=False,c='gist_rainbow',N_type='single',ts_factor=0.6,folder=None):
    
    fig, axs= plt.subplots(nrows=2, ncols=2, sharex= True, sharey= False)

    ax=axs.flatten()
    L=[]
    A=[]


    lab={-1:'Clean',7:'Gaussian',8:'Colored',9:'Linear',10:'Random (fft)',11:'Hyperbolic',12:'Bandpassed',
        130: 'Lowpass', 14:'Trace-wise',13: 'Lowpass',
        0:'A: Gaussian + Color', 
        1:'B: A + Linear',
        2:'C: D + Gaussian(f-k)',
        3:'E: C + Hyperbolic',
        6:'F: B + Trace-wise',
        4:'G: B + Lowpass', 
        5:'H: G + Trace-wise', 
        15:'A + Gaussain (f-k)', 
        16:'Gaussian (x-t)+(f-k)', 
        17:'Linear + Hyperbolic', 
        18: 'Colored + Linear',
        19: 'Gaussian + Linear'}
    
    if folder is not None: 
        folder =folder
    else:
        folder=None
                
    
    if N_type=='single':
        N=[7,10,9,11,8,12,13,14]

    else:
        if model_type =='restormer':
            N=[16,15,0,19,18,17,1,2,3,4,6,5]
        else:
            N=[16,15,0,18,17,1,2,3,4,6,5]

    R=len(N)
        
    if N_type=='single':
            
        c=['b','darkorange','limegreen','blueviolet','r','sienna','hotpink','dimgray']
    else:
        if model_type =='restormer':
            c=['r','darkorange','b','darkgreen','limegreen','c','k','lavender','blueviolet','hotpink','dimgray','sienna']
        else:   
            c=['r','darkorange','b','limegreen','c','k','lavender','blueviolet','hotpink','dimgray','sienna']
            
    for problem in ['firstbreak','denoise']:
        j=0
    
        log0 = read_logs(model_type = model_type, noise_type=-1, noise_scale=0.0,problem=problem,attack=attack,
                        pretrained=pretrained,folder=None).ewm(alpha=(1 - ts_factor), ignore_na=True).mean()
        if problem=='firstbreak':
                k=0
        else:
                k=2
        ax[0+k].plot(log0[log0.columns[0]],log0[log0.columns[1]],'-',label=lab[-1],color='y')
        ax[1+k].plot(log0[log0.columns[0]],log0[log0.columns[2]],'-',label=lab[-1],color='y')
        for noise_type in N:
            DF=pd.DataFrame(columns=[np.arange(8)])
            DF2=pd.DataFrame(columns=[np.arange(8)])
            i=0
            for noise_scale in [0.05,0.1,0.15,0.2,0.25,0.5,1.0,2.0]: 
                    
                try:
                    log = read_logs(model_type = model_type, noise_type=noise_type, 
                                    noise_scale=noise_scale,problem=problem,
                                attack=attack,pretrained=pretrained,
                                folder=folder)
                except FileNotFoundError:
                    continue
                DF[DF.columns[i]]=(log[log.columns[1]].ewm(alpha=(1 - ts_factor), ignore_na=True).mean())
                DF2[DF2.columns[i]]=(log[log.columns[2]].ewm(alpha=(1 - ts_factor), ignore_na=True).mean())
                
                i+=1

            M=np.nanmean(DF,axis=1)
            
            M2=np.nanmean(DF2,axis=1)
            

            fac=1

            ax[0+k].plot(smooth2(np.arange(0,100,1),fac),smooth2(M,fac),'-',label=lab[noise_type],color=c[j])
     
            ax[1+k].plot(smooth2(np.arange(0,100,1),fac),smooth2(M2,fac),'-',label=lab[noise_type],color=c[j])

            j+=1

    
        C_noise = mpatches.Patch(color='w',linewidth=0,label='')
        

        h,l = ax[0].get_legend_handles_labels()
        if N_type == 'single':
            L=ax[0].legend([h[0],C_noise, h[1],h[2], h[3], h[4], h[5],h[6], h[7],h[8]],#h[9]],
                            ['Clean','','Gaussian','Gaussian(f-k)','Linear','Hyperbolic','Colored','Bandpassed','Lowpass','Trace-wise'],#'LP_ran'],
                                loc='upper center', bbox_to_anchor=(1.2, -1.5),
                                fontsize=12,
                                labelcolor='k',facecolor='none',edgecolor='none',ncol=5)
            
        else: 
            if model_type=="restormer":
                L=ax[0].legend([h[0],h[1],h[2],h[3], h[4], h[5],h[6], h[7], h[8], h[9], h[10],h[11],h[12]],
                                ['Clean','Gaussian + Gaussian(f-k)',
                                'Gaussian + Gaussian(f-k) + Colored',
                                'Gaussian + Colored',
                                'Gaussian + Linear',
                                'Colored + Linear',
                                'Linear + Hyperbolic', 
                                'X: Gaussian + Colored + Linear',
                                'X + Gaussian(f-k)',
                                'X + Gaussian(f-k) + Hyperbolic',
                                'X + Lowpass (fixed $freq.$)', 'X + Trace-wise', 
                                'X + Lowpass (fixed $freq.$) + Trace-wise'],
                                    loc='upper center', bbox_to_anchor=(1.2, -1.5),
                                    fontsize=12,
                                    labelcolor='k',facecolor='none',edgecolor='none',ncol=3)  
            else:
                L=ax[0].legend([h[0],h[1],h[2],h[3], h[4], h[5],h[6], h[7], h[8], h[9], h[10],h[11]],
                                ['Clean','Gaussian + Gaussian(f-k)',
                                'Gaussian + Gaussian(f-k) + Colored',
                                'Gaussian + Colored',
                                'Colored + Linear',
                                'Linear + Hyperbolic', 
                                'X: Gaussian + Colored + Linear',
                                'X + Gaussian(f-k)',
                                'X + Gaussian(f-k) + Hyperbolic',
                                'X + Lowpass (fixed $freq.$)', 'X + Trace-wise', 
                                'X + Lowpass (fixed $freq.$) + Trace-wise'],
                                    loc='upper center', bbox_to_anchor=(1.2, -1.5),
                                    fontsize=12,
                                    labelcolor='k',facecolor='none',edgecolor='none',ncol=3)  


        for b in range (0,R+1,1):
            L.get_lines()[b].set_linewidth(3)
            L.get_lines()[b].set_alpha(1)

        for i in [1,3]:
            ax[i].locator_params(axis ='y',nbins=5)
            ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                
        plt.subplots_adjust(right=2)
        for i in [0,2]:
            ax[i].set_ylabel('Loss',fontsize=14, fontweight='bold')
            if model_type != 'unet':
                if N_type =='single':
                    ax[0].set_ylim(0,0.05)
                else:
                    if model_type=='restormer':
                        ax[0].set_ylim(0,0.03)
                    else:
                        ax[0].set_ylim(0,0.08)
            else:
                ax[0].set_ylim(0.3,0.42)
        
            ax[i].locator_params(axis ='y',nbins=4)
            ax[i].ticklabel_format(axis ='y',style='sci',scilimits=(0,0))
        for i in [2,3]:
            ax[i].set_xlabel('Epoch',fontsize=14, fontweight='bold')
            ax[i].set_xlim(1,85)
            if model_type != 'unet':
                if N_type =='single':
                    ax[2].set_ylim(0,0.015)
                else:
                    if model_type=='restormer':
                        ax[2].set_ylim(0,0.01)
                    else:
                        ax[2].set_ylim(0,0.03)
            else:
                ax[2].set_ylim(0,0.18)



        if problem =='firstbreak':

            ax[1].set_ylabel('Accuracy (mIoU)',fontsize=14, fontweight='bold')
            if N_type =='single':
                if model_type != 'unet':
                    ax[1].set_ylim(0.96,1)
                else:
                    ax[1].set_ylim(0.97,1)
            else:
                if model_type != 'swin':
                    ax[1].set_ylim(0.97,1)
                else:
                    ax[1].set_ylim(0.96,1)
        else:

            ax[3].set_ylabel('Accuracy (RMSE)',fontsize=14, fontweight='bold')
            if model_type != 'unet':
                if model_type =='restormer':
                    ax[i].set_ylim(0.1,0)
                else:
                    ax[i].set_ylim(0.15,0)
            else:
                ax[i].set_ylim(0.3,0.1)

    if N_type== 'single':
        if model_type != 'unet':
            ax[2].text(10,-0.007, 'Single Noise:', fontsize=12, fontweight='bold')
            ax[0].text(-15, 0.025, '(i)', fontsize=14, fontweight='bold')   
            ax[2].text(-15, 0.007, '(ii)', fontsize=14, fontweight='bold') 
        else:
            ax[2].text(10,-0.085, 'Single Noise:', fontsize=12, fontweight='bold')
            ax[0].text(-15, 0.36, '(i)', fontsize=14, fontweight='bold')   
            ax[2].text(-15, 0.08, '(ii)', fontsize=14, fontweight='bold') 
    
    else:
        if model_type != 'unet':
            if model_type == 'restormer':
                ax[2].text(-12,-0.0025, 'Compound Noise:', fontsize=12, fontweight='bold')
                ax[0].text(-15, 0.015, '(i)', fontsize=14, fontweight='bold')   
                ax[2].text(-15, 0.005, '(ii)', fontsize=14, fontweight='bold')
            else:
                ax[2].text(-12,-0.013, 'Compound Noise:', fontsize=12, fontweight='bold')
                ax[0].text(-15, 0.04, '(i)', fontsize=14, fontweight='bold')   
                ax[2].text(-15, 0.015, '(ii)', fontsize=14, fontweight='bold')
            
        else:
            ax[2].text(-12,-0.12, 'Compound Noise:', fontsize=12, fontweight='bold')
            ax[0].text(-15, 0.37, '(i)', fontsize=14, fontweight='bold')   
            ax[2].text(-15, 0.08, '(ii)', fontsize=14, fontweight='bold')


def smooth2(y, box_pts):
    if box_pts ==0: 
        y_smooth =y
    else:
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
    return y_smooth