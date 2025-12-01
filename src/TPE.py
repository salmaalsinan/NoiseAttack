import optuna
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys, os, argparse, warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Your local imports
sys.path.append("../src/models/")
sys.path.append("../src/")
from models.build import build_model
from noiseadding import build_noise_transforms
from data import get_train_val_dataset, get_dataset

# --------------- One-time setup ---------------
torch.backends.cudnn.benchmark = True

def make_loaders(train_dataset, val_dataset, bs):
    return (
        DataLoader(train_dataset, batch_size=bs, shuffle=True,
                   num_workers=4, pin_memory=True, persistent_workers=True),
        DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                   num_workers=4, pin_memory=True, persistent_workers=True),
    )

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total = 0.0
    for sample in loader:
        x = sample["input"].float().to(device, non_blocking=True)
        y = sample["target"]
        # cast once based on task
        y = y.long() if criterion.__class__.__name__ == "CrossEntropyLoss" else y
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for sample in loader:
        x = sample["input"].float().to(device, non_blocking=True)
        y = sample["target"]
        y = y.long() if criterion.__class__.__name__ == "CrossEntropyLoss" else y
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def make_objective(model_type, problem, device, train_dataset, val_dataset,
                   max_epochs, tune_batch_size):
    def objective(trial: optuna.Trial):
        # --- search space
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        if tune_batch_size:
            bs = trial.suggest_categorical("batch_size", [8, 16, 32])
        else:
            bs = 8

        train_loader, val_loader = make_loaders(train_dataset, val_dataset, bs)

        model = build_model(model_type, problem).to(device)
        if problem == "denoise":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        patience = 3
        best, bad = float("inf"), 0
        for epoch in range(max_epochs):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss = validate(model, val_loader, criterion, device)

            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss + 1e-6 < best:
                best, bad = val_loss, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        return best
    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)         # e.g. 'unet'/'swin'/'restormer'
    parser.add_argument("--problem", type=str, required=True)       # 'firstbreak' or 'denoise'
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--noise_type", type=int, default=-1)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--tune_batch_size", action="store_true")   # <-- flag to tune BS
    parser.add_argument("--n_trials", type=int, default=40)
    args = parser.parse_args()

    model_type = args.model
    problem = args.problem

    if args.tune_batch_size:
        prefix='lr_bs'
    else:
        prefix='lr'

    # device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # datasets
    noise_transforms = build_noise_transforms(noise_type=args.noise_type, scale=args.noise_scale)
    full_dataset = get_dataset(problem, noise_transforms=noise_transforms)
    train_dataset, val_dataset = get_train_val_dataset(full_dataset)

    # study / storage
    os.makedirs("../random/TPE/", exist_ok=True)
    storage = f"sqlite:///../random/TPE/tpe_{prefix}_{model_type}_{problem}_{args.prefix}.db"

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, constant_liar=True)
    pruner  = optuna.pruners.HyperbandPruner(
        min_resource=1, #5
        max_resource=args.epochs,
        reduction_factor=3, #2 
        bootstrap_count=1, #5
    )
    
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=f"seismic_{prefix}_{model_type}_{problem}",
        load_if_exists=True,
    )

    objective = make_objective(
        model_type=model_type,
        problem=problem,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_epochs=args.epochs,
        tune_batch_size=args.tune_batch_size,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True, n_jobs=1)

    print("Best:", study.best_value)
    print("Params:", study.best_params)

    # Save trial table
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "distributions"))
    out_csv = f"../random/TPE/tpe_study_summary_{prefix}_{model_type}_{problem}_{args.prefix}.csv"
    df.to_csv(out_csv, index=False)
    with open(out_csv, "a") as f:
        f.write("\n# Best value:,{}\n".format(study.best_value))
        for k, v in study.best_params.items():
            f.write(f"\n# {k}: {v}\n")

    print(f"Saved study summary to {out_csv}")

    #LR
    #python TPE.py --model unet --problem firstbreak --epochs 12 --n_trials 40 
    #LR+BS
    #python TPE.py --model unet --problem firstbreak --epochs 12 --n_trials 40 --tune_batch_size
