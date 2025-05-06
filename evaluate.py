import torch
from torch.utils.data import DataLoader
from model import RawNet
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train
from core_scripts.startup_config import set_random_seed
import yaml
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

# ---- Config ----
PROTOCOL_PATH = 'D:/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
FLAC_DIR = 'D:/evals/typing/'
YAML_PATH = 'model_config_RawNet.yaml'
MODEL_PATH = 'models\model_LA_CCE_11_32_0.0001_AE Trained\epoch_9.pth'
BATCH_SIZE = 32
SEED = 1234

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_random_seed(SEED, None)

# Load config and model
with open(YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)
model = RawNet(config['model'], device).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("Model loaded.")

# Load data
d_label, file_list = genSpoof_list(PROTOCOL_PATH, is_train=False, is_eval=False)
dataset = Dataset_ASVspoof2019_train(file_list, d_label, base_dir=FLAC_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get raw model scores
model.eval()
y_true, y_score = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        y_true.extend(y.view(-1).tolist())
        output = model(x)
        score = output[:, 1].cpu().numpy()  # spoof logits
        y_score.extend(score)

y_true = np.array(y_true)
y_score = np.array(y_score)

# Flip score polarity so that higher means spoof
y_score = -y_score

# ---- Threshold Search ----
best_acc = 0.0
best_thresh = None
thresholds = np.linspace(-5.0, 1.0, 1000)

for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    acc = (y_pred == y_true).mean()
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

# Evaluate at best threshold
y_pred = (y_score >= best_thresh).astype(int)
acc = (y_pred == y_true).mean()
cm = confusion_matrix(y_true, y_pred)

# EER
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, pos_label=1)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

# ---- Output ----
print(f"\nBest Threshold: {best_thresh:.4f}")
print(f"Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:")
print(cm)
print(f"EER: {eer * 100:.2f}%")




