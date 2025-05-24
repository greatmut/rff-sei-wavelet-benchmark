import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from models_baseline_and_custom import WaveletCNN_8, SmallScalogramCNN, ResNet18Baseline

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

BASE_DATASET_ROOT = os.path.abspath("track_A_wavelet_scalograms_png_channel3")

EXPERIMENTS = [
    "AWGN_0", "AWGN_5", "AWGN_10", "AWGN_15", "AWGN_20", "AWGN_25", "AWGN_30",
    "Rician_0", "Rician_5", "Rician_10", "Rician_15", "Rician_30",
    "Doppler_0", "Doppler_50", "Doppler_100", "Doppler_200", "Doppler_400", "Doppler_800",
    "Combined_Average"
]
WAVELET_DIRS = [
    "pywt_Morlet", "pywt_Mexh", "pywt_Gaus8",
    "ssq_GMW", "ssq_Bump", "ssq_CMHat",
    "ssq_SST_GMW", "ssq_SST_Bump", "ssq_SST_HHat"
]
TX_NAMES = ["Tx01", "Tx02", "Tx03", "Tx04", "Tx05", "Tx06", "Tx07", "Tx08", "Tx09", "Tx10", "Tx11"]
BATCH_SIZE = 256
IMG_SIZE = 128
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
VAL_SPLIT = 0.15
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transforms(img_size, is_train=True, is_rgb=False):
    tf = [
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor()
    ]
    if is_train:
        tf = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0), antialias=True)
        ] + tf
    if not is_rgb:
        tf = [T.Grayscale(num_output_channels=1)] + tf
    return T.Compose(tf)

def plot_auc_roc(all_labels, all_probs, num_classes, result_dir, model_name="Model"):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    y_true = label_binarize(all_labels, classes=np.arange(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        if np.sum(y_true[:, i]) == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    if y_true.shape[1] > 1:
        fpr["macro"], tpr["macro"], _ = roc_curve(y_true.ravel(), all_probs.ravel())
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"], tpr["macro"], roc_auc["macro"] = [None], [None], None

    plt.figure(figsize=(8, 6))
    if "macro" in fpr and fpr["macro"][0] is not None:
        plt.plot(fpr["macro"], tpr["macro"], label='macro-average (AUC = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    for i in range(num_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "auc_roc.png"))
    plt.close()

def plot_tsne(features, labels, result_dir, model_name="Model"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=3000)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{model_name} t-SNE of Test Set Features")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "tsne.png"))
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 7), dpi=400)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', linewidths=1.5,
                cbar=False, square=True,
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Device', fontsize=16, weight='bold')
    plt.xlabel('Predicted Device', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close()

def evaluate_model(model, loader, class_names, log_dir, wavelet_name, is_rgb):
    model.eval()
    y_true, y_pred, feats = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, 1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(lbls.numpy())
            y_pred.extend(preds)
            feats.append(probs)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    feats = np.concatenate(feats, axis=0)
    print("\nClassification Report:\n")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(log_dir, f"{wavelet_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    plot_tsne(feats, y_true, log_dir, wavelet_name)
    plot_auc_roc(y_true, feats, len(class_names), log_dir, wavelet_name)
    print(f"Saved plots: {cm_path}, tsne.png, auc_roc.png")
    try:
        aucs = roc_auc_score(np.eye(len(class_names))[y_true], feats, average=None, multi_class="ovr")
        for i, aucv in enumerate(aucs):
            print(f"ROC AUC for class {class_names[i]}: {aucv:.4f}")
    except Exception as e:
        print(f"Could not compute ROC AUC: {e}")

def run_training(model, model_name, train_loader, val_loader, test_loader, class_names, log_dir, wavelet_name, in_channels=1, is_rgb=False):
    summary(model, input_size=(BATCH_SIZE, in_channels, IMG_SIZE, IMG_SIZE))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    patience_cnt = 0

    start_time = time.time()
    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- Training started at: {start_str} ---\n")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = 0, 0
        total_train = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, 1)
            train_acc += (preds == lbls).sum().item()
            total_train += imgs.size(0)
        train_loss /= total_train
        train_acc /= total_train

        model.eval()
        val_loss, val_acc = 0, 0
        total_val = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, 1)
                val_acc += (preds == lbls).sum().item()
                total_val += imgs.size(0)
        val_loss /= total_val
        val_acc /= total_val

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(log_dir, f"best_model_{model_name}.pt")
            torch.save(model.state_dict(), best_model_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered.")
                break

    end_time = time.time()
    end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    print(f"\n--- Training ended at: {end_str} ---")
    print(f"--- Training duration: {duration/60:.2f} minutes ({duration:.2f} seconds) ---\n")
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")
    evaluate_model(model, test_loader, class_names, log_dir, wavelet_name, is_rgb)
    print(f"\nAll results, model summary, and training logs saved to {log_dir}\n")

if __name__ == "__main__":
    for experiment in EXPERIMENTS:
        for wavelet_name in WAVELET_DIRS:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # --- WaveletCNN_8 Section (Grayscale) ---
            log_dir_waveletcnn = f"results_{experiment}_{wavelet_name}_{now}_WaveletCNN8"
            os.makedirs(log_dir_waveletcnn, exist_ok=True)
            print(f"\n====== Experiment: {experiment} | Wavelet: {wavelet_name} | Model: WaveletCNN_8 ======")
            train_dir = os.path.join(BASE_DATASET_ROOT, experiment, wavelet_name, "train")
            test_dir = os.path.join(BASE_DATASET_ROOT, experiment, wavelet_name, "test")
            if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
                print(f"Skipping {wavelet_name} in {experiment}: No train/test dirs at {train_dir} or {test_dir}.\n")
                continue
            if not any(os.listdir(train_dir)) or not any(os.listdir(test_dir)):
                print(f"Skipping {wavelet_name} in {experiment}: Empty train/test dir.\n")
                continue
            train_ds = ImageFolder(train_dir, transform=get_transforms(IMG_SIZE, True, is_rgb=False))
            test_ds = ImageFolder(test_dir, transform=get_transforms(IMG_SIZE, False, is_rgb=False))
            n_train = len(train_ds)
            n_val = int(VAL_SPLIT * n_train)
            n_train = n_train - n_val
            train_set, val_set = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_STATE))
            class_names = TX_NAMES
            num_classes = len(class_names)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            model_waveletcnn = WaveletCNN_8(num_classes).to(DEVICE)
            run_training(model_waveletcnn, "WaveletCNN8", train_loader, val_loader, test_loader, class_names, log_dir_waveletcnn, wavelet_name, in_channels=1, is_rgb=False)

            # --- SmallScalogramCNN Section (RGB) ---
            log_dir_smallcnn = f"results_{experiment}_{wavelet_name}_{now}_SmallScalogramCNN"
            os.makedirs(log_dir_smallcnn, exist_ok=True)
            print(f"\n====== Experiment: {experiment} | Wavelet: {wavelet_name} | Model: SmallScalogramCNN ======")
            train_ds_smallcnn = ImageFolder(train_dir, transform=get_transforms(IMG_SIZE, True, is_rgb=True))
            test_ds_smallcnn = ImageFolder(test_dir, transform=get_transforms(IMG_SIZE, False, is_rgb=True))
            n_train_smallcnn = len(train_ds_smallcnn)
            n_val_smallcnn = int(VAL_SPLIT * n_train_smallcnn)
            n_train_smallcnn = n_train_smallcnn - n_val_smallcnn
            train_set_smallcnn, val_set_smallcnn = random_split(train_ds_smallcnn, [n_train_smallcnn, n_val_smallcnn], generator=torch.Generator().manual_seed(RANDOM_STATE))
            train_loader_smallcnn = DataLoader(train_set_smallcnn, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            val_loader_smallcnn = DataLoader(val_set_smallcnn, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            test_loader_smallcnn = DataLoader(test_ds_smallcnn, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            model_smallcnn = SmallScalogramCNN(num_classes).to(DEVICE)
            run_training(model_smallcnn, "SmallScalogramCNN", train_loader_smallcnn, val_loader_smallcnn, test_loader_smallcnn, class_names, log_dir_smallcnn, wavelet_name, in_channels=3, is_rgb=True)

            # --- ResNet18 Section (RGB) ---
            log_dir_resnet = f"results_{experiment}_{wavelet_name}_{now}_ResNet18"
            os.makedirs(log_dir_resnet, exist_ok=True)
            print(f"\n====== Experiment: {experiment} | Wavelet: {wavelet_name} | Model: ResNet18 ======")
            train_ds_resnet = ImageFolder(train_dir, transform=get_transforms(IMG_SIZE, True, is_rgb=True))
            test_ds_resnet = ImageFolder(test_dir, transform=get_transforms(IMG_SIZE, False, is_rgb=True))
            n_train_res = len(train_ds_resnet)
            n_val_res = int(VAL_SPLIT * n_train_res)
            n_train_res = n_train_res - n_val_res
            train_set_res, val_set_res = random_split(train_ds_resnet, [n_train_res, n_val_res], generator=torch.Generator().manual_seed(RANDOM_STATE))
            train_loader_resnet = DataLoader(train_set_res, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            val_loader_resnet = DataLoader(val_set_res, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            test_loader_resnet = DataLoader(test_ds_resnet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            resnet18 = ResNet18Baseline(num_classes=num_classes, pretrained=True, finetune_last_blocks_only=True).to(DEVICE)
            run_training(resnet18, "ResNet18", train_loader_resnet, val_loader_resnet, test_loader_resnet, class_names, log_dir_resnet, wavelet_name, in_channels=3, is_rgb=True)