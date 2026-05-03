# src/main_benchmark_task1_task2.py
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from dataset_api import EEGLongitudinalDataset, get_dataloaders
from models.eegnet import EEGNet
from models.shallowcnn import ShallowConvNet
from models.deepcnn import DeepConvNet
from models.eegconformer import Conformer
from models.dgcnn import DGCNN

def resolve_path(path_value, fallback_base_dir):
    if os.path.isabs(path_value): return path_value
    cwd_candidate = os.path.abspath(path_value)
    if os.path.exists(cwd_candidate): return cwd_candidate
    return os.path.abspath(os.path.join(fallback_base_dir, path_value))

def load_config(config_path="../config/task1_config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = resolve_path(config_path, script_dir)
    with open(resolved_path, "r") as f:
        return yaml.safe_load(f), resolved_path

def build_model(model_name, config):
    params = config['models'][model_name]
    nc, cl, sig = config['data']['num_channels'], config['data']['num_classes'], config['data']['sig_length']

    if model_name == "EEGNet":
        return EEGNet(nb_classes=cl, number_channel=nc, signal_length=sig,
                      kernel_size=params['kernel_size'], f1=params['f1'], D=params['D'],
                      f2=params['f1']*params['D'], dropout_rate=params['dropout_rate'])
    elif model_name == "Conformer":
        return Conformer(nb_classes=cl, number_channel=nc,
                         emb_size=params['emb_size'], num_heads=params['num_heads'], depth=params['depth'],
                         dropout_rate=params['dropout_rate'], time_kernel=params['time_kernel'])
    elif model_name == "ShallowConvNet":
        return ShallowConvNet(nb_classes=cl, number_channel=nc,
                              dropout_rate=params['dropout_rate'], time_kernel=params['time_kernel'])
    elif model_name == "DeepConvNet":
        return DeepConvNet(nb_classes=cl, number_channel=nc,
                           dropout_rate=params['dropout_rate'], time_kernel=params['time_kernel'])
    elif model_name == "DGCNN":
        return DGCNN(in_channels=sig, num_electrodes=nc,
                     k_adj=params.get('k_adj', 2),
                     out_channels=params.get('out_channels', 32),
                     num_classes=cl)

def evaluate_model(model, dataloader, device, num_classes=2):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.unsqueeze(1).to(device), y.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            if num_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.cpu().numpy())

    present_classes = set(np.unique(all_labels))
    metrics = {
        'acc':  accuracy_score(all_labels, all_preds),
        'bacc': balanced_accuracy_score(all_labels, all_preds),
        'f1':   f1_score(all_labels, all_preds, average='macro', labels=list(range(num_classes)),
                         zero_division=0),
    }
    if num_classes == 2:
        try:
            metrics['auc']    = roc_auc_score(all_labels, all_probs)
            metrics['pr_auc'] = average_precision_score(all_labels, all_probs)
        except ValueError:
            metrics['auc'] = metrics['pr_auc'] = float('nan')
    else:
        try:
            all_probs_arr = np.array(all_probs)
            # Pad prob matrix to num_classes columns when some classes are absent
            if all_probs_arr.shape[1] < num_classes:
                pad = np.zeros((all_probs_arr.shape[0], num_classes - all_probs_arr.shape[1]))
                all_probs_arr = np.hstack([all_probs_arr, pad])
            metrics['auc']    = roc_auc_score(all_labels, all_probs_arr, multi_class='ovr',
                                              average='macro', labels=list(range(num_classes)))
            y_true_bin        = label_binarize(all_labels, classes=list(range(num_classes)))
            metrics['pr_auc'] = average_precision_score(y_true_bin, all_probs_arr, average='macro')
        except ValueError as e:
            print(f"  [WARN] AUC/PR-AUC skipped (not all classes present): {e}")
            metrics['auc'] = metrics['pr_auc'] = float('nan')

    return metrics, all_preds, all_labels, all_probs

def main(args):
    config, resolved_config_path = load_config(args.config)
    config_dir = os.path.dirname(resolved_config_path)

    data_root = resolve_path(config['data']['data_root'], config_dir)
    tsv_path  = resolve_path(config['data']['tsv_path'],  config_dir)
    config['data']['data_root'] = data_root
    config['data']['tsv_path']  = tsv_path

    # --epochs CLI flag overrides config value (useful for quick sanity checks)
    if args.epochs is not None:
        config['experiment']['epochs']  = args.epochs
        config['experiment']['patience'] = args.epochs

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['data']['num_classes']

    run_name = f"T{args.task}_{args.model}_{args.strategy}"
    if args.group != "ALL": run_name += f"_{args.group}"
    if args.task == 2:      run_name += f"_{args.session}"
    if args.subject:        run_name += f"_sub-{args.subject}"
    if args.fold:           run_name += f"_fold-{args.fold}"

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"Device : {device}")
    print(f"Epochs : {config['experiment']['epochs']}  |  Patience : {config['experiment']['patience']}")
    print(f"{'='*60}")

    if args.strategy.startswith('intra'):
        window_sec  = config['data'].get('window_sec',  4.0)
        overlap_sec = config['data'].get('overlap_sec', 3.5)
    else:
        window_sec  = config['data'].get('window_sec',  4.0)
        overlap_sec = config['data'].get('overlap_sec', 2.0)

    config['data']['sig_length'] = int(window_sec * 250.0)

    group_val = None if args.group == "ALL" else args.group
    dataset = EEGLongitudinalDataset(
        data_root, tsv_path,
        task_id=args.task,
        group_filter=group_val,
        task2_session=args.session,
        window_sec=window_sec,
        overlap_sec=overlap_sec
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, cv_strategy=args.strategy, test_subject=args.subject,
        batch_size=config['experiment']['batch_size'],
        random_seed=args.fold if args.fold else 42
    )

    model     = build_model(args.model, config).to(device)
    params    = config['models'][args.model]
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_bacc   = 0.0
    patience_counter = 0
    checkpoint_dir  = os.path.join("results", "checkpoints", f"task{args.task}", args.model, args.strategy)
    if args.group != "ALL":  checkpoint_dir = os.path.join(checkpoint_dir, args.group)
    if args.subject:         checkpoint_dir = os.path.join(checkpoint_dir, f"subject_{args.subject.replace('sub-', '')}")
    elif args.fold is not None: checkpoint_dir = os.path.join(checkpoint_dir, f"fold_{args.fold}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_{run_name}.pth")

    for epoch in range(config['experiment']['epochs']):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.unsqueeze(1).to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, num_classes)
        print(f"  Epoch {epoch+1:>3}/{config['experiment']['epochs']} | "
              f"Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | Val BAcc: {val_metrics['bacc']:.4f}")

        if val_metrics['bacc'] > best_val_bacc:
            best_val_bacc = val_metrics['bacc']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config['experiment']['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics, test_preds, test_labels, _ = evaluate_model(model, test_loader, device, num_classes)

    print(f"\n{'='*60}")
    print(f"RESULT  {run_name}")
    print(f"  Acc    : {test_metrics['acc']:.4f}")
    print(f"  BAcc   : {test_metrics['bacc']:.4f}")
    print(f"  F1     : {test_metrics['f1']:.4f}")
    print(f"  AUC    : {test_metrics['auc']:.4f}")
    print(f"  PR-AUC : {test_metrics['pr_auc']:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True,
                        choices=["EEGNet", "ShallowConvNet", "DeepConvNet", "Conformer", "DGCNN"])
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["inter", "loso", "intra_block", "intra_chrono"])
    parser.add_argument("--subject",  type=str, default=None)
    parser.add_argument("--fold",     type=int, default=None)
    parser.add_argument("--config",   type=str, default="../config/task1_config.yaml")
    parser.add_argument("--task",     type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--group",    type=str, default="ALL", choices=["ALL", "HK", "SA", "BF"])
    parser.add_argument("--session",  type=str, default="pre", choices=["pre", "post"])
    parser.add_argument("--epochs",   type=int, default=None,
                        help="Override config epochs (and patience). Useful for quick sanity checks.")
    args = parser.parse_args()
    main(args)
