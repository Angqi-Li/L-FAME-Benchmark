import os
import glob
import numpy as np
import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import re
import warnings


# ==========================================
# 1) Dataset: indexing, labeling, and loading mode
# ==========================================
class EEGLongitudinalDataset(Dataset):
    """
    Window-based EEG dataset for longitudinal meditation benchmark tasks.

    Purpose:
    - Scan pre-exported continuous `.npy` EEG files
    - Apply task-specific inclusion and labeling rules
    - Build a global window index for window-level sampling
    """

    def __init__(self, data_root, tsv_path, task_id, preprocessing_desc='preproc',
                 group_filter=None, task2_session='pre',
                 window_sec=4.0, overlap_sec=2.0, sfreq=250.0, preload_to_ram=True):
        """
        Initialize dataset configuration and build global window index.
        """
        
        # Validate window overlap configuration.
        if overlap_sec >= window_sec:
            raise ValueError(
                f"Invalid arguments: overlap_sec ({overlap_sec}) must be smaller than window_sec ({window_sec})."
            )
            
        self.data_root = data_root
        self.task_id = task_id
        self.preprocessing_desc = preprocessing_desc
        self.group_filter = group_filter.upper() if group_filter is not None else None
        self.task2_session = task2_session.lower()
        self.sfreq = sfreq
        self.window_pts = int(window_sec * sfreq)
        self.stride_pts = int((window_sec - overlap_sec) * sfreq)
        
        # Loading mode and file-level caches.
        self.preload_to_ram = preload_to_ram
        self.ram_cache = {}
        self.mmap_cache = {}

        if self.group_filter is not None and self.group_filter not in {"HK", "SA", "BF"}:
            raise ValueError(
                f"Invalid group_filter: {group_filter}. Allowed values are HK, SA, BF."
            )
        if self.task2_session not in {"pre", "post"}:
            raise ValueError(
                f"Invalid task2_session: {task2_session}. Allowed values are pre, post."
            )
        
        # Subject-level metadata
        self.subject_info = self._load_participants(tsv_path)
        self.global_index = []
        
        search_pattern = os.path.join(data_root, '**', f'*_eeg_{preprocessing_desc}.npy')
        file_paths = glob.glob(search_pattern, recursive=True)
        
        group_msg = self.group_filter if self.group_filter is not None else "ALL"
        print(f"[INFO] Initialize Dataset | Task {task_id} | Preprocessing: {preprocessing_desc} "
              f"| Group: {group_msg} | Scanned files: {len(file_paths)}")

        # Task 3: keep only subjects with both pre and post slMedita sessions.
        paired_subjects = None
        if task_id == 3:
            subject_sessions = {}
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                if 'task-slMedita' not in filename: continue

                parts = filename.split('_')
                sub_str = next((p for p in parts if p.startswith('sub-')), None)
                ses_str = next((p for p in parts if p.startswith('ses-')), None)
                if not sub_str or not ses_str: continue

                sub_id = str(int(sub_str.replace('sub-', '')))
                if sub_id not in subject_sessions: subject_sessions[sub_id] = set()
                if ses_str in {'ses-premedita', 'ses-posmedita'}: subject_sessions[sub_id].add(ses_str)

            paired_subjects = {
                sub_id for sub_id, sessions in subject_sessions.items()
                if {'ses-premedita', 'ses-posmedita'}.issubset(sessions)
            }
            print(f"   -> Task 3 paired-subject filter: retained {len(paired_subjects)} subjects with both pre and post.")
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            sub_str = next((p for p in parts if p.startswith('sub-')), None)
            ses_str = next((p for p in parts if p.startswith('ses-')), None)
            
            if not sub_str or not ses_str:
                continue
                
            sub_id = str(int(sub_str.replace('sub-', '')))
            subject_group = self.subject_info.get(sub_id, None)

            if self.group_filter is not None and subject_group != self.group_filter:
                continue
            
            label = None
            
            if task_id == 1:
                if ses_str == 'ses-premedita':
                    if 'task-restCE01' in filename: label = 0
                    elif 'task-slMedita' in filename: label = 1

            elif task_id == 2:
                target_session = 'ses-premedita' if self.task2_session == 'pre' else 'ses-posmedita'
                if ses_str == target_session:
                    if 'task-slMedita' in filename or 'task-Medita' in filename:
                        group = subject_group
                        if group == 'HK': label = 0
                        elif group == 'SA': label = 1
                        elif group == 'BF': label = 2

            elif task_id == 3:
                if paired_subjects is not None and sub_id not in paired_subjects: 
                    continue
                
                if ses_str == 'ses-posmedita':
                    if 'task-restCE01' in filename: 
                        label = 0
                    elif 'task-slMedita' in filename: 
                        label = 1
            
            if label is None:
                continue
                
            # Data access strategy:
            # - preload_to_ram=True: load each file once into RAM.
            # - preload_to_ram=False: inspect metadata with mmap and load slices lazily.
            if self.preload_to_ram:
                if file_path not in self.ram_cache:
                    # Convert to float32 to reduce memory usage.
                    self.ram_cache[file_path] = np.load(file_path).astype(np.float32)
                n_times = self.ram_cache[file_path].shape[1]
            else:
                # Read only array shape via mmap, without full materialization.
                temp_mmap = np.load(file_path, mmap_mode='r')
                n_times = temp_mmap.shape[1]
            
            for start_idx in range(0, n_times - self.window_pts + 1, self.stride_pts):
                self.global_index.append({
                    'file_path': file_path,
                    'start_idx': start_idx,
                    'end_idx': start_idx + self.window_pts,
                    'label': label,
                    'subject': sub_id
                })

        mode_str = "RAM" if self.preload_to_ram else "Disk (mmap)"
        print(f"[INFO] Dataset index built: {len(self.global_index)} windows via {mode_str}.")

    def _load_participants(self, tsv_path):
        df = pd.read_csv(tsv_path, sep='\t')
        sub_dict = {}
        for _, row in df.iterrows():
            raw_id = str(row['participant_id']).strip()
            clean_id = str(int(raw_id.replace('sub-', '')))
            sub_dict[clean_id] = str(row['group']).strip().upper()
        return sub_dict

    def __len__(self):
        return len(self.global_index)

    # def __getitem__(self, idx):
    #     record = self.global_index[idx]
    #     file_path = record['file_path']
    #     start_idx = record['start_idx']
    #     end_idx = record['end_idx']
        
    #     # Fast path when data is preloaded in memory.
    #     if self.preload_to_ram:
    #         slice_data = self.ram_cache[file_path][:, start_idx:end_idx]
    #         X = torch.tensor(slice_data, dtype=torch.float32)
    #     else:
    #         if file_path not in self.mmap_cache:
    #             self.mmap_cache[file_path] = np.load(file_path, mmap_mode='r')
    #         slice_data = self.mmap_cache[file_path][:, start_idx:end_idx]
    #         X = torch.tensor(slice_data, dtype=torch.float32)
            
    #     y = torch.tensor(record['label'], dtype=torch.long)
    #     return X, y
    def __getitem__(self, idx):
        record = self.global_index[idx]
        file_path = record['file_path']
        start_idx = record['start_idx']
        end_idx = record['end_idx']
        
        # read original data
        if self.preload_to_ram:
            slice_data = self.ram_cache[file_path][:, start_idx:end_idx]
            X = torch.tensor(slice_data, dtype=torch.float32)
        else:
            if file_path not in self.mmap_cache:
                self.mmap_cache[file_path] = np.load(file_path, mmap_mode='r')
            slice_data = self.mmap_cache[file_path][:, start_idx:end_idx]
            X = torch.tensor(slice_data, dtype=torch.float32)
            

        X_mean = X.mean(dim=1, keepdim=True)
        X_std = X.std(dim=1, keepdim=True)
        # add 1e-6 to prevent NaN when standard deviation is 0 (e.g. when bad channels are filled with zeros)
        X = (X - X_mean) / (X_std + 1e-6)
        
        # return normalized data and label
        y = torch.tensor(record['label'], dtype=torch.long)
        return X, y


# ==========================================
# 2) DataLoader builder
# ==========================================
def get_dataloaders(
    dataset,
    cv_strategy,
    test_subject=None,
    batch_size=64,
    random_seed=42,
    use_weighted_sampler=True,
):
    if dataset.task_id == 2 and cv_strategy.startswith('intra'):
        raise ValueError("Task 2 does not support intra-subject strategies.")

    subjects_array = np.array([record['subject'] for record in dataset.global_index])
    labels_array = np.array([record['label'] for record in dataset.global_index])
    unique_subjects = np.unique(subjects_array)

    train_idx, test_idx = [], []
    safety_gap = math.ceil(dataset.window_pts / dataset.stride_pts)

    def normalize_subject_to_int(subject_token):
        token = str(subject_token).replace("sub-", "")
        if not token.isdigit(): return None
        return int(token)

    subject_ids_num = np.array(
        [normalize_subject_to_int(s) if normalize_subject_to_int(s) is not None else -1 for s in subjects_array]
    )

    def split_chrono(indices):
        split_point = int(len(indices) * 0.8)
        return indices[:split_point], indices[split_point + safety_gap:]

    def split_block(indices):
        local_train, local_test = [], []
        block_pts = 20.0 * dataset.sfreq
        block_size = max(1, math.ceil(block_pts / dataset.stride_pts))
        pattern = ['train', 'train', 'train', 'train', 'test']
        num_epochs = len(indices)
        num_blocks = math.ceil(num_epochs / block_size)
        last_assignment = None

        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, num_epochs)
            current_assignment = pattern[i % len(pattern)]
            if last_assignment is not None and current_assignment != last_assignment:
                start_idx += safety_gap
            if start_idx >= end_idx: continue
            
            block_indices = indices[start_idx:end_idx]
            if current_assignment == 'train': local_train.extend(block_indices)
            else: local_test.extend(block_indices)
            last_assignment = current_assignment
        return local_train, local_test

    if cv_strategy == 'loso':
        target_sub_num = normalize_subject_to_int(test_subject)
        test_sub_clean = test_subject.replace('sub-', '').zfill(2)
        print(f"[INFO] LOSO | Hold-out subject: sub-{test_sub_clean}")
        train_idx = np.where(subject_ids_num != target_sub_num)[0].tolist()
        test_idx = np.where(subject_ids_num == target_sub_num)[0].tolist()

    elif cv_strategy == 'inter':
        if len(unique_subjects) < 2:
            raise ValueError(f"Not enough subjects ({len(unique_subjects)}) for inter-subject split. Need at least 2.")
            
        # Task 2 classifies by meditation group (HK/SA/BF), so split subjects with
        # stratification to keep group proportions balanced in train and test.
        if dataset.task_id == 2:
            subject_labels = []
            for sub in unique_subjects:
                # Look up the group label from the first window of this subject.
                idx = np.where(subjects_array == sub)[0][0]
                subject_labels.append(labels_array[idx])

            try:
                train_subs, test_subs = train_test_split(
                    unique_subjects, test_size=0.2, random_state=random_seed, stratify=subject_labels
                )
            except ValueError:
                # Fallback for small datasets where stratified split cannot place ≥1 subject
                # per class in the test fold (e.g. quickstart subset with 3 subjects/group).
                print("[WARN] Stratified subject split failed (too few subjects per class); "
                      "falling back to random split.")
                train_subs, test_subs = train_test_split(
                    unique_subjects, test_size=0.2, random_state=random_seed
                )
        else:
            # Task 1 / Task 3: each subject has both class labels, so a simple random
            # subject-level split is sufficient (no need for group stratification).
            train_subs, test_subs = train_test_split(
                unique_subjects, test_size=0.2, random_state=random_seed
            )
            
        train_idx = np.where(np.isin(subjects_array, train_subs))[0].tolist()
        test_idx = np.where(np.isin(subjects_array, test_subs))[0].tolist()

    elif cv_strategy.startswith('intra'):
        target_sub_num = normalize_subject_to_int(test_subject)
        subj_indices = np.where(subject_ids_num == target_sub_num)[0].tolist()

        if dataset.task_id == 3:
            # ==========================================
            # Task 3: Cross-Session Adaptation (Zero-shot / Few-shot)
            # ==========================================
            subj_label0 = [idx for idx in subj_indices if labels_array[idx] == 0] # Post: Rest
            subj_label1 = [idx for idx in subj_indices if labels_array[idx] == 1] # Post: Meditation
            
            # if cv_strategy contains 'zero' or 'few', use zero-shot or few-shot mode
            if 'zero' in cv_strategy:
                print("[INFO] Task 3: Zero-Shot Generalization Mode")
                # Zero-shot: all Post data are used for Test, no Train
                train_idx = []
                test_idx = subj_label0 + subj_label1
                
            elif 'shot' in cv_strategy:
                # use 'intra_5_shot' or '10shot' etc. to extract number from string
                match = re.search(r'(\d+)[_-]?shot', cv_strategy)
                
                if match:
                    num_shots = int(match.group(1))
                    print(f"[INFO] Task 3: {num_shots}-Shot Calibration Mode")
                else:
                    # if 'few_shot' is passed but no number is written, keep default value 30 (approximately 60 seconds)
                    num_shots = 30
                    print(f"[WARN] No specific shot number found, defaulting to {num_shots}-Shot Calibration Mode")
                
                # split label 0 (Rest)
                l0_train = subj_label0[:num_shots]
                # important: add safety_gap between Few-shot train and test to prevent leakage!
                l0_test = subj_label0[num_shots + safety_gap:] if len(subj_label0) > (num_shots + safety_gap) else []
                
                # split label 1 (Meditation)
                l1_train = subj_label1[:num_shots]
                l1_test = subj_label1[num_shots + safety_gap:] if len(subj_label1) > (num_shots + safety_gap) else []
                
                train_idx = l0_train + l1_train
                test_idx = l0_test + l1_test
                
            elif 'chrono' in cv_strategy:
                print("[INFO] Task 3: Full Post-Session Intra-Subject (Chrono Split)")
                l0_train, l0_test = split_chrono(subj_label0)
                l1_train, l1_test = split_chrono(subj_label1)
                train_idx = l0_train + l1_train
                test_idx  = l0_test  + l1_test

            elif 'block' in cv_strategy:
                print("[INFO] Task 3: Full Post-Session Intra-Subject (Block Split)")
                l0_train, l0_test = split_block(subj_label0)
                l1_train, l1_test = split_block(subj_label1)
                train_idx = l0_train + l1_train
                test_idx  = l0_test  + l1_test

            else:
                raise ValueError("For Task 3, cv_strategy must contain 'zero', 'shot', 'chrono', or 'block'.")

        else:
            # Task 1 intra-subject: split within a single subject's pre-session data.
            subj_label0 = [idx for idx in subj_indices if labels_array[idx] == 0]
            subj_label1 = [idx for idx in subj_indices if labels_array[idx] == 1]
            
            if 'chrono' in cv_strategy:
                l0_train, l0_test = split_chrono(subj_label0)
                l1_train, l1_test = split_chrono(subj_label1)
            else:
                l0_train, l0_test = split_block(subj_label0)
                l1_train, l1_test = split_block(subj_label1)
                
            train_idx = l0_train + l1_train
            test_idx = l0_test + l1_test
    else:
        raise ValueError(f"Unsupported cv_strategy: {cv_strategy}")

    is_zero_shot = (dataset.task_id == 3 and 'zero' in cv_strategy)

    # ==========================================
    # Carve a validation split out of train_idx
    # ==========================================
    val_idx = []

    # Skip val for zero-shot (no train data) and few-shot (too little data).
    # chrono/block on Task 3 use full post data, so val is available.
    skip_val = is_zero_shot or (dataset.task_id == 3 and 'shot' in cv_strategy)
    if not skip_val and len(train_idx) > 0:
        rng = np.random.RandomState(random_seed)
        train_idx_arr = np.array(train_idx)

        if cv_strategy in ('loso', 'inter'):
            # subject-level val: hold out training subjects so val mirrors test distribution
            train_subs_arr = subjects_array[train_idx_arr]
            unique_train_subs = np.unique(train_subs_arr)

            sub_to_label = {}
            for s in unique_train_subs:
                first_local = np.where(train_subs_arr == s)[0][0]
                sub_to_label[s] = labels_array[train_idx_arr[first_local]]

            if cv_strategy == 'loso':
                # 1 subject (task 1/3) or 1-per-class (task 2)
                if dataset.task_id == 2 and len(unique_train_subs) >= 3:
                    val_subs = []
                    for lab in np.unique(list(sub_to_label.values())):
                        cands = [s for s, l in sub_to_label.items() if l == lab]
                        val_subs.append(rng.choice(cands))
                else:
                    val_subs = [rng.choice(unique_train_subs)] if len(unique_train_subs) > 1 else []
            else:
                # inter: hold out ~20% of training subjects, stratified by label for task 2
                if dataset.task_id == 2:
                    sub_arr = np.array(list(sub_to_label.keys()))
                    lab_arr = np.array(list(sub_to_label.values()))
                    try:
                        _, val_subs = train_test_split(
                            sub_arr, test_size=0.2, random_state=random_seed, stratify=lab_arr
                        )
                        val_subs = val_subs.tolist()
                    except ValueError:
                        # fallback when a class has too few subjects to stratify
                        val_subs = [rng.choice(unique_train_subs)]
                else:
                    n_val = max(1, int(round(len(unique_train_subs) * 0.2)))
                    val_subs = rng.choice(unique_train_subs, size=n_val, replace=False).tolist()

            if len(val_subs) > 0:
                val_mask = np.isin(train_subs_arr, val_subs)
                val_idx = train_idx_arr[val_mask].tolist()
                train_idx = train_idx_arr[~val_mask].tolist()
                print(f"[INFO] {cv_strategy.upper()} subject-level val | val subjects: {list(val_subs)} | "
                      f"train win: {len(train_idx)} | val win: {len(val_idx)}")
        else:
            # intra*: only one subject in train pool, fall back to window-level stratified split
            train_labels_arr = labels_array[train_idx_arr]
            stratify = train_labels_arr if len(np.unique(train_labels_arr)) > 1 else None
            new_train, val_idx = train_test_split(
                train_idx, test_size=0.1, random_state=random_seed, stratify=stratify
            )
            train_idx = list(new_train)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx) if len(val_idx) > 0 else None
    test_dataset = Subset(dataset, test_idx)

    if len(train_dataset) == 0 and not is_zero_shot:
        print(f"[WARN] Train dataset is empty! cv_strategy={cv_strategy}, test_subject={test_subject}")
    if len(test_dataset) == 0:
        print(f"[WARN] Test dataset is empty! cv_strategy={cv_strategy}, test_subject={test_subject}")

    # DataLoader settings for throughput.
    loader_kwargs = {
        'num_workers': 4,
        'pin_memory': True,          # Speed up host-to-device transfers.
        'prefetch_factor': 2,        # Prefetch batches to reduce pipeline stalls.
        'persistent_workers': True   # Keep workers alive across epochs.
    }

    if is_zero_shot:
        train_loader = None  # Zero-shot: no fine-tuning, so no train loader is needed.
    else:
        if use_weighted_sampler:
            train_labels = np.array([dataset.global_index[idx]['label'] for idx in train_idx], dtype=int)
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            class_weight = {label: 1.0 / count for label, count in zip(unique_labels, counts)}
            sample_weights = np.array([class_weight[label] for label in train_labels], dtype=np.float64)
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **loader_kwargs)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    if not is_zero_shot:
        train_label_dist = np.bincount(np.array([dataset.global_index[i]['label'] for i in train_idx], dtype=int))
        print(f"   -> Train labels: {dict(enumerate(train_label_dist))}")
        if val_loader is not None:
            val_label_dist = np.bincount(np.array([dataset.global_index[i]['label'] for i in val_idx], dtype=int))
            print(f"   -> Val labels:   {dict(enumerate(val_label_dist))}")

    test_label_dist = np.bincount(np.array([dataset.global_index[i]['label'] for i in test_idx], dtype=int))
    print(f"   -> Test labels:  {dict(enumerate(test_label_dist))}\n")

    return train_loader, val_loader, test_loader
