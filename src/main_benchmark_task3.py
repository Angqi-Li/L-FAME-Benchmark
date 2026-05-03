# src/main_benchmark_task3.py
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import torch.nn.functional as F

from dataset_api import EEGLongitudinalDataset, get_dataloaders
from main_benchmark_task1_task2 import load_config, build_model, evaluate_model, resolve_path

def find_task1_checkpoint(base_dir, model_name, subject_id, intra_strategy='intra_chrono'):
    """
    Locate the per-subject Task 1 checkpoint under
        <base_dir>/<model_name>/<intra_strategy>/subject_<XX>/best_*.pth
    """
    if intra_strategy not in ('intra_chrono', 'intra_block'):
        raise ValueError(f"intra_strategy must be 'intra_chrono' or 'intra_block', got: {intra_strategy}")

    raw = str(subject_id).replace("sub-", "")
    sub_variants = {raw, raw.zfill(2), raw.lstrip('0') or '0'}

    candidates = []
    for sv in sub_variants:
        subject_dir = os.path.join(base_dir, model_name, intra_strategy, f"subject_{sv}")
        if os.path.isdir(subject_dir):
            candidates.extend(glob.glob(os.path.join(subject_dir, "best_*.pth")))

    if not candidates:
        for sv in sub_variants:
            pattern = os.path.join(base_dir, model_name, intra_strategy, '**', f'*sub-{sv}*.pth')
            candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find Task 1 {intra_strategy} checkpoint for {model_name} sub-{raw} under {base_dir}"
        )

    best_checkpoint = max(candidates, key=os.path.getmtime)
    print(f"[INFO] Discovered Task 1 Checkpoint: {best_checkpoint}")
    return best_checkpoint

def fine_tune_model(model, train_loader, val_loader, device, num_classes, config, ft_mode='full', lr=5e-5):
    if ft_mode == 'linear':
        print("[INFO] Fine-tuning mode: LINEAR (freezing feature extractor)")
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name and 'dense' not in name.lower():
                param.requires_grad = False
    else:
        print("[INFO] Fine-tuning mode: FULL (updating all parameters)")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epochs    = config['experiment'].get('ft_epochs', 20)

    best_val_bacc   = 0.0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.unsqueeze(1).to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if val_loader:
            val_metrics, _, _, _ = evaluate_model(model, val_loader, device, num_classes)
            print(f"  FT Epoch {epoch+1:>2}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Val BAcc: {val_metrics['bacc']:.4f}")
            if val_metrics['bacc'] > best_val_bacc:
                best_val_bacc   = val_metrics['bacc']
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            print(f"  FT Epoch {epoch+1:>2}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def main(args):
    config, resolved_config_path = load_config(args.config)
    config_dir   = os.path.dirname(resolved_config_path)

    data_root    = resolve_path(config['data']['data_root'], config_dir)
    tsv_path     = resolve_path(config['data']['tsv_path'],  config_dir)
    task1_log_dir = resolve_path(args.task1_dir, config_dir)

    # --ft_epochs CLI flag overrides config value
    if args.ft_epochs is not None:
        config['experiment']['ft_epochs'] = args.ft_epochs

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['data']['num_classes']

    run_name = f"T3_{args.model}_{args.strategy}_sub-{args.subject}"
    if 'zero' not in args.strategy:
        run_name += f"_FT-{args.ft_mode}"

    print(f"\n{'='*60}")
    print(f"Run    : {run_name}")
    print(f"Device : {device}")
    if 'zero' not in args.strategy:
        print(f"FT epochs : {config['experiment'].get('ft_epochs', 20)}")
    print(f"{'='*60}")

    window_sec  = config['data'].get('window_sec',  4.0)
    overlap_sec = config['data'].get('overlap_sec', 3.5)
    config['data']['sig_length'] = int(window_sec * 250.0)

    dataset = EEGLongitudinalDataset(
        data_root, tsv_path,
        task_id=3,
        group_filter=None,
        window_sec=window_sec,
        overlap_sec=overlap_sec
    )

    present_subs = {r['subject'] for r in dataset.global_index}
    raw_sub  = args.subject.replace('sub-', '')
    try:
        clean_sub = str(int(raw_sub))
    except ValueError:
        clean_sub = raw_sub

    if clean_sub not in present_subs:
        print(f"[SKIP] sub-{raw_sub} has no post-meditation data in Task 3 paired set.")
        return

    strategy_for_loader = args.strategy if args.strategy.startswith('intra') else f"intra_{args.strategy}"
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, cv_strategy=strategy_for_loader, test_subject=args.subject,
        batch_size=config['experiment']['batch_size'],
        random_seed=42
    )

    model           = build_model(args.model, config)
    checkpoint_path = find_task1_checkpoint(task1_log_dir, args.model, args.subject,
                                            intra_strategy=args.task1_intra)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device)

    if 'zero' in args.strategy:
        print("[INFO] Zero-shot evaluation — bypassing fine-tuning.")
    else:
        print(f"[INFO] Few-shot calibration. Train batches: {len(train_loader)}")
        ft_lr = config['models'][args.model].get('ft_lr', 5e-5)
        model = fine_tune_model(
            model, train_loader, val_loader, device, num_classes,
            config, ft_mode=args.ft_mode, lr=ft_lr
        )

        ft_ckpt_dir = os.path.join(
            "results", "checkpoints", "task3", args.model, args.strategy,
            f"subject_{clean_sub.zfill(2)}"
        )
        os.makedirs(ft_ckpt_dir, exist_ok=True)
        ft_ckpt_path = os.path.join(ft_ckpt_dir, f"best_{run_name}.pth")
        torch.save(model.state_dict(), ft_ckpt_path)
        print(f"  Checkpoint saved: {ft_ckpt_path}")

    test_metrics, _, _, _ = evaluate_model(model, test_loader, device, num_classes)

    print(f"\n{'='*60}")
    print(f"RESULT  {run_name}")
    print(f"  Acc  : {test_metrics['acc']:.4f}")
    print(f"  BAcc : {test_metrics['bacc']:.4f}")
    print(f"  F1   : {test_metrics['f1']:.4f}")
    print(f"  AUC  : {test_metrics['auc']:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: Longitudinal Cross-Session Generalization")
    parser.add_argument("--model",      type=str, required=True,
                        choices=["EEGNet", "ShallowConvNet", "DeepConvNet", "Conformer", "DGCNN"])
    parser.add_argument("--strategy",   type=str, required=True,
                        help="E.g. 'zero_shot', '10_shot'")
    parser.add_argument("--subject",    type=str, required=True,
                        help="Target subject e.g. 'sub-01'")
    parser.add_argument("--config",     type=str, default="../config/task3_config.yaml")
    parser.add_argument("--task1_dir",  type=str, default="../results/checkpoints/task1")
    parser.add_argument("--task1_intra",type=str, default="intra_chrono",
                        choices=["intra_chrono", "intra_block"])
    parser.add_argument("--ft_mode",    type=str, default="full", choices=["full", "linear"])
    parser.add_argument("--ft_epochs",  type=int, default=None,
                        help="Override config ft_epochs (useful for quick sanity checks).")
    args = parser.parse_args()
    main(args)
