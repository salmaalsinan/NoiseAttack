import os
import pandas as pd
import glob
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import argparse
from pathlib import Path


def _load_run(path):
  event_acc = event_accumulator.EventAccumulator(path)
  event_acc.Reload()
  data = {}

  for tag in sorted(event_acc.Tags()["scalars"]):
    x, y = [], []

    for scalar_event in event_acc.Scalars(tag):
      x.append(scalar_event.step)
      y.append(scalar_event.value)

    data[tag] = (np.asarray(x), np.asarray(y))
  return data



def FILE(file):

    if file.sort(key=os.path.getmtime, reverse=True)[0]==file.sort(key=os.path.getsize, reverse=True)[0]:
        file.sort(key=os.path.getmtime, reverse=True)
    else:
        file.sort(key=os.path.getsize, reverse=True)
    print(file)
    return file
    
def Get_CSV(logdir,outdir):
  t = os.listdir(logdir)
  t.sort()
  trials=t[:] #change based on how many you want to exclude from the sorted main folder

  for i in range(0,len(trials)):
      e=[]
      l=[]
      m=[]
      f=f'{logdir}/{trials[i]}/'
      ID=trials[i]
      print(ID)
      name=ID.split('_')[0]
      trials2 = glob.glob(f'{f}/*')

      trials2.sort(key=os.path.getsize, reverse=True)

      trials3 = trials2[0]
      d=_load_run(trials3)
      print(d.keys())
      if ID.split('_')[1] =='denoise':
          metric='RMSE'
      else:
          metric='mIoU'
      e=d['Loss'][0]
      l=d['Loss'][1]
      m=d[metric][1]
      M=np.vstack([e,l,m])
      np.shape(M.T)
      df=pd.DataFrame(M.T,columns=['Epoch','Loss',f'{metric}'])
      df.to_csv(f'{outdir}/{ID}.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=False, default='firstbreak')
    parser.add_argument("--logdir", type=str, required=False, default=None)
    parser.add_argument("--outdir", type=str, required=False, default=None)
    args = parser.parse_args()

    rootdir = Path("../tensorboard")   # <- using Path

    # LOGDIR ----------------------------------------------------
    if args.logdir is None:
        logdir = rootdir / "logs" / args.problem
    else:
        logdir = Path(args.logdir)

    logdir.mkdir(parents=True, exist_ok=True)

    # OUTDIR ----------------------------------------------------
    if args.outdir is None:
        outdir = rootdir / "csv" / args.problem
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # call your function with resolved paths
    Get_CSV(str(logdir), str(outdir))

    print(f"All CSV files are saved in {outdir}")

  

