import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import json
import warnings
import time  # ADICIONADO

warnings.filterwarnings('ignore')

class Config:
    train_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original'
    train_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth'
    test_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Original'
    test_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth'
    
    num_classes = 1
    img_size = 224
    batch_size = 4
    epochs = 50
    learning_rate = 0.0001
    
    n_runs = 5
    save_results = True
    results_dir = './results'
    model_name = 'SwinUNet'
    experiment_name = f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    patience = 10
    min_delta = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = './best_SwinUNet_segmentation.pth'
    
    num_workers = 2
    pin_memory = True if torch.cuda.is_available() else False
    pretrained = True
    
    # NOVA CONFIGURAÇÃO: Métricas de tempo
    measure_time = True

config = Config()

os.makedirs(config.results_dir, exist_ok=True)
network_dir = os.path.join(config.results_dir, config.model_name)
os.makedirs(network_dir, exist_ok=True)
experiment_dir = os.path.join(network_dir, config.experiment_name)
os.makedirs(experiment_dir, exist_ok=True)
reports_dir = os.path.join(network_dir, f'RELATORIO_DAS_EXECUCOES')
os.makedirs(reports_dir, exist_ok=True)

print(f"✅ Diretórios criados:")
print(f"   Network: {network_dir}")
print(f"   Experimento: {experiment_dir}")
print(f"   Relatórios: {reports_dir}")
print(f"   Device: {config.device}")
print()

def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    else:
        return obj

class SwinUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinUNet, self).__init__()
        self.num_classes = num_classes

        if pretrained:
            try:
                self.swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            except:
                self.swin = models.swin_t(weights='IMAGENET1K_V1')
        else:
            self.swin = models.swin_t(weights=None)

        self.patch_embed = self.swin.features[0]
        self.stage1 = self.swin.features[1]
        self.stage2 = self.swin.features[2]
        self.stage3 = self.swin.features[3]
        self.stage4 = self.swin.features[4]

        self._detect_channels()
        self._build_decoder()

    def _detect_channels(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            out = self.patch_embed(x)
            if out.dim() == 4 and out.shape[1] > out.shape[-1] and out.shape[1] > 100:
                self.patch_format = 'BCHW'
                self.c1 = out.shape[1]
                out = out.permute(0, 2, 3, 1)
            else:
                self.patch_format = 'BHWC'
                self.c1 = out.shape[-1]
            print(f"📌 PatchEmbed output format: {self.patch_format}, channels: {self.c1}")

            x = self.stage1(out)
            self.c2 = x.shape[-1]
            x = self.stage2(x)
            self.c3 = x.shape[-1]
            x = self.stage3(x)
            self.c4 = x.shape[-1]
            x = self.stage4(x)
            self.c5 = x.shape[-1]

        print(f"📊 Canais detectados: c1={self.c1}, c2={self.c2}, c3={self.c3}, c4={self.c4}, c5={self.c5}")

    def _build_decoder(self):
        self.up4 = nn.Conv2d(self.c5, self.c4, kernel_size=3, padding=1)
        self.up3 = nn.Conv2d(self.c4, self.c3, kernel_size=3, padding=1)
        self.up2 = nn.Conv2d(self.c3, self.c2, kernel_size=3, padding=1)
        self.up1 = nn.Conv2d(self.c2, self.c1, kernel_size=3, padding=1)

        self.dec4 = nn.Sequential(
            nn.Conv2d(self.c4 + self.c4, self.c4, 3, padding=1),
            nn.BatchNorm2d(self.c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c4, self.c4, 3, padding=1),
            nn.BatchNorm2d(self.c4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.c3 + self.c3, self.c3, 3, padding=1),
            nn.BatchNorm2d(self.c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c3, self.c3, 3, padding=1),
            nn.BatchNorm2d(self.c3),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.c2 + self.c2, self.c2, 3, padding=1),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2, self.c2, 3, padding=1),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.c1 + self.c1, self.c1, 3, padding=1),
            nn.BatchNorm2d(self.c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c1, self.c1, 3, padding=1),
            nn.BatchNorm2d(self.c1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(self.c1, self.num_classes, kernel_size=1)

    def forward(self, x):
        out = self.patch_embed(x)
        if self.patch_format == 'BCHW':
            out = out.permute(0, 2, 3, 1)

        e1 = self.stage1(out)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)

        e1_bchw = e1.permute(0, 3, 1, 2)
        e2_bchw = e2.permute(0, 3, 1, 2)
        e3_bchw = e3.permute(0, 3, 1, 2)
        e4_bchw = e4.permute(0, 3, 1, 2)

        d4 = nn.functional.interpolate(e4_bchw, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = self.up4(d4)
        if d4.shape[2:] != e3_bchw.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e3_bchw.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3_bchw], dim=1)
        d4 = self.dec4(d4)

        d3 = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.up3(d3)
        if d3.shape[2:] != e2_bchw.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e2_bchw.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2_bchw], dim=1)
        d3 = self.dec3(d3)

        d2 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up2(d2)
        if d2.shape[2:] != e1_bchw.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1_bchw.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1_bchw], dim=1)
        d2 = self.dec2(d2)

        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up1(d1)
        d1 = nn.functional.interpolate(d1, size=(224, 224), mode='bilinear', align_corners=False)

        out = self.final_conv(d1)
        return out

# ============================================
# DATASET
# ============================================
class FundusSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*.*g')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*.*g')))
        print(f"📂 Imagens: {len(self.images_paths)}, Máscaras: {len(self.masks_paths)}")

        img_names = {os.path.basename(p).lower(): p for p in self.images_paths}
        mask_names = {os.path.basename(p).lower(): p for p in self.masks_paths}

        self.valid_pairs = []
        for name, img_path in img_names.items():
            if name in mask_names:
                self.valid_pairs.append((img_path, mask_names[name]))
            else:
                base = os.path.splitext(name)[0]
                for mname, mpath in mask_names.items():
                    if os.path.splitext(mname)[0] == base:
                        self.valid_pairs.append((img_path, mpath))
                        break

        print(f"✅ Pares válidos: {len(self.valid_pairs)}")
        if len(self.valid_pairs) == 0:
            raise ValueError("Nenhum par encontrado!")
        self.img_size = img_size

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

# ============================================
# LOSS E MÉTRICAS
# ============================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        preds_sigmoid = torch.sigmoid(preds)
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        return bce + dice_loss

def compute_metrics(preds, targets, threshold=0.5):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if preds.min() < 0 or preds.max() > 1:
        preds = 1 / (1 + np.exp(-preds))

    preds_binary = (preds > threshold).astype(np.uint8)
    targets_binary = targets.astype(np.uint8)

    acc = accuracy_score(targets_binary.flatten(), preds_binary.flatten())
    intersection = (preds_binary & targets_binary).sum()
    dice = (2.0 * intersection + 1e-6) / (preds_binary.sum() + targets_binary.sum() + 1e-6)
    union = (preds_binary | targets_binary).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    sens = (intersection + 1e-6) / (targets_binary.sum() + 1e-6)
    tn = ((1 - preds_binary) & (1 - targets_binary)).sum()
    spec = (tn + 1e-6) / ((1 - targets_binary).sum() + 1e-6)
    return {
        'accuracy': float(acc),
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sens),
        'specificity': float(spec)
    }

# ============================================
# FUNÇÕES DE TREINAMENTO COM MEDIÇÃO DE TEMPO
# ============================================
def train_epoch(model, loader, criterion, optimizer, device, measure_time=True):
    model.train()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    epoch_start_time = time.time()
    batch_times = []
    data_load_times = []
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        if measure_time:
            data_load_start = time.time()
        
        images, masks = images.to(device), masks.to(device)
        
        if measure_time:
            data_load_times.append(time.time() - data_load_start)
            forward_start = time.time()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        if measure_time:
            batch_times.append(time.time() - forward_start)
        
        running_loss += loss.item()

        batch_metrics = compute_metrics(outputs.detach(), masks.detach())
        for k in metrics:
            metrics[k] += batch_metrics[k]
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    num_batches = len(loader)
    avg_loss = running_loss / num_batches
    for k in metrics:
        metrics[k] /= num_batches
    
    time_metrics = {}
    if measure_time and batch_times:
        time_metrics = {
            'epoch_total_time': time.time() - epoch_start_time,
            'batch_forward_time_mean': np.mean(batch_times),
            'batch_forward_time_std': np.std(batch_times),
            'batch_forward_time_min': np.min(batch_times),
            'batch_forward_time_max': np.max(batch_times),
            'data_load_time_mean': np.mean(data_load_times) if data_load_times else 0,
            'data_load_time_total': np.sum(data_load_times) if data_load_times else 0,
            'batches_per_second': num_batches / (time.time() - epoch_start_time)
        }
    
    return avg_loss, metrics, time_metrics

def validate_epoch(model, loader, criterion, device, measure_time=True):
    model.eval()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    epoch_start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            if measure_time:
                start_time = time.time()
            
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            if measure_time:
                inference_times.append(time.time() - start_time)
            
            running_loss += loss.item()
            batch_metrics = compute_metrics(outputs, masks)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    
    num_batches = len(loader)
    avg_loss = running_loss / num_batches
    for k in metrics:
        metrics[k] /= num_batches
    
    time_metrics = {}
    if measure_time and inference_times:
        time_metrics = {
            'epoch_total_time': time.time() - epoch_start_time,
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'inference_time_min': np.min(inference_times),
            'inference_time_max': np.max(inference_times),
            'inferences_per_second': num_batches / (time.time() - epoch_start_time)
        }
    
    return avg_loss, metrics, time_metrics

# ============================================
# FUNÇÃO DE TREINAMENTO COM EARLY STOPPING E TEMPO
# ============================================
def train_model(model, train_loader, val_loader, config, run_id=0):
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    stopped_epoch = config.epochs

    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_metrics': [], 'val_metrics': [],
        'train_time': [], 'val_time': []
    }
    
    training_start_time = time.time()

    print(f"\n🚀 Run {run_id+1}/{config.n_runs} - {config.model_name}")
    print(f"Device: {config.device}, Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Paciência: {config.patience}, Épocas: {config.epochs}")
    print(f"⏱️ Medição de tempo: {'Ativada' if config.measure_time else 'Desativada'}\n")

    for epoch in range(config.epochs):
        print(f"\n{'='*50}\nRun {run_id+1} - Época {epoch+1}/{config.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_metrics, train_time = train_epoch(
            model, train_loader, criterion, optimizer, config.device, config.measure_time
        )
        val_loss, val_metrics, val_time = validate_epoch(
            model, val_loader, criterion, config.device, config.measure_time
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)

        print(f"📊 Treino - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"📊 Validação - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        if config.measure_time:
            print(f"⏱️  Treino: {train_time.get('epoch_total_time', 0):.2f}s | "
                  f"Batch: {train_time.get('batch_forward_time_mean', 0):.3f}s")
            print(f"⏱️  Validação: {val_time.get('epoch_total_time', 0):.2f}s | "
                  f"Inferência: {val_time.get('inference_time_mean', 0):.3f}s")

        if val_metrics['dice'] > best_dice + config.min_delta:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"{config.best_model_path}_run{run_id}")
            print(f"✅ Melhor modelo salvo! Dice: {best_dice:.4f} (época {epoch+1})")
        else:
            patience_counter += 1
            print(f"⏳ Paciência: {patience_counter}/{config.patience} (melhor Dice: {best_dice:.4f} na época {best_epoch+1})")

        if patience_counter >= config.patience:
            stopped_epoch = epoch + 1
            print(f"\n🛑 Early stopping na época {stopped_epoch}")
            break
    
    total_training_time = time.time() - training_start_time

    history['early_stop'] = {
        'stopped_epoch': stopped_epoch,
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'patience_used': patience_counter >= config.patience,
        'total_training_time': total_training_time
    }

    if config.save_results:
        save_run_results(history, run_id, config)

    return history, best_dice

# ============================================
# SALVAR RESULTADOS
# ============================================
def save_run_results(history, run_id, config):
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)

    data = {
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_dice': history['train_dice'],
        'val_dice': history['val_dice'],
    }
    
    # Adicionar métricas de tempo
    if history['train_time'] and isinstance(history['train_time'][0], dict):
        for key in history['train_time'][0].keys():
            data[f'train_time_{key}'] = [t.get(key, 0) if isinstance(t, dict) else 0 for t in history['train_time']]
    
    if history['val_time'] and isinstance(history['val_time'][0], dict):
        for key in history['val_time'][0].keys():
            data[f'val_time_{key}'] = [t.get(key, 0) if isinstance(t, dict) else 0 for t in history['val_time']]
    
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        data[f'train_{metric}'] = [m[metric] for m in history['train_metrics']]
        data[f'val_{metric}'] = [m[metric] for m in history['val_metrics']]

    pd.DataFrame(data).to_csv(os.path.join(run_dir, 'training_results.csv'), index=False)

    final_metrics = {
        'run_id': int(run_id),
        'model_name': config.model_name,
        'img_size': config.img_size,
        'best_val_dice': float(max(history['val_dice'])),
        'final_val_dice': float(history['val_dice'][-1]),
        'final_train_dice': float(history['train_dice'][-1]),
        'best_val_loss': float(min(history['val_loss'])),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_train_loss': float(history['train_loss'][-1]),
        'best_epoch': int(np.argmax(history['val_dice']) + 1),
        'total_epochs': len(history['train_loss']),
        'early_stopped': history['early_stop']['patience_used'],
        'stopped_epoch': history['early_stop']['stopped_epoch'],
        'total_training_time': float(history['early_stop'].get('total_training_time', 0)),
    }
    
    # Adicionar métricas de tempo finais
    if history['train_time'] and isinstance(history['train_time'][-1], dict):
        for key, value in history['train_time'][-1].items():
            final_metrics[f'final_train_{key}'] = float(value) if isinstance(value, (int, float)) else 0
    
    if history['val_time'] and isinstance(history['val_time'][-1], dict):
        for key, value in history['val_time'][-1].items():
            final_metrics[f'final_val_{key}'] = float(value) if isinstance(value, (int, float)) else 0
    
    # Média das métricas de tempo por época
    train_time_values = [t.get('epoch_total_time', 0) for t in history['train_time'] if isinstance(t, dict)]
    val_time_values = [t.get('epoch_total_time', 0) for t in history['val_time'] if isinstance(t, dict)]
    
    if train_time_values:
        final_metrics['avg_train_epoch_time'] = float(np.mean(train_time_values))
        final_metrics['std_train_epoch_time'] = float(np.std(train_time_values))
    
    if val_time_values:
        final_metrics['avg_val_epoch_time'] = float(np.mean(val_time_values))
        final_metrics['std_val_epoch_time'] = float(np.std(val_time_values))
    
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        final_metrics[f'final_val_{metric}'] = float(history['val_metrics'][-1][metric])
        final_metrics[f'final_train_{metric}'] = float(history['train_metrics'][-1][metric])

    with open(os.path.join(run_dir, 'final_metrics.json'), 'w') as f:
        json.dump(convert_to_serializable(final_metrics), f, indent=4)

    plot_training_history(history, run_id, config)
    plot_time_metrics(history, run_id, config)

def plot_training_history(history, run_id, config):
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{config.model_name} - Run {run_id+1}', fontsize=16, fontweight='bold')

    axes[0,0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0,0].plot(history['val_loss'], label='Val Loss', marker='s')
    if history['early_stop']['patience_used']:
        axes[0,0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss'); axes[0,0].legend(); axes[0,0].grid()

    axes[0,1].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[0,1].plot(history['val_dice'], label='Val Dice', marker='s')
    if history['early_stop']['patience_used']:
        axes[0,1].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Dice'); axes[0,1].legend(); axes[0,1].grid()

    train_iou = [m['iou'] for m in history['train_metrics']]
    val_iou = [m['iou'] for m in history['val_metrics']]
    axes[1,0].plot(train_iou, label='Train IoU', marker='o')
    axes[1,0].plot(val_iou, label='Val IoU', marker='s')
    if history['early_stop']['patience_used']:
        axes[1,0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('IoU'); axes[1,0].legend(); axes[1,0].grid()

    train_acc = [m['accuracy'] for m in history['train_metrics']]
    val_acc = [m['accuracy'] for m in history['val_metrics']]
    axes[1,1].plot(train_acc, label='Train Accuracy', marker='o')
    axes[1,1].plot(val_acc, label='Val Accuracy', marker='s')
    if history['early_stop']['patience_used']:
        axes[1,1].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Accuracy'); axes[1,1].legend(); axes[1,1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_plots.png'), dpi=300)
    plt.close()

def plot_time_metrics(history, run_id, config):
    """Plota e salva os gráficos de métricas de tempo"""
    if not config.measure_time:
        return
    
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    
    # Extrair métricas de tempo
    train_epoch_times = []
    val_epoch_times = []
    train_batch_times = []
    val_inference_times = []
    
    for train_t, val_t in zip(history['train_time'], history['val_time']):
        if isinstance(train_t, dict):
            train_epoch_times.append(train_t.get('epoch_total_time', 0))
            train_batch_times.append(train_t.get('batch_forward_time_mean', 0))
        else:
            train_epoch_times.append(0)
            train_batch_times.append(0)
        
        if isinstance(val_t, dict):
            val_epoch_times.append(val_t.get('epoch_total_time', 0))
            val_inference_times.append(val_t.get('inference_time_mean', 0))
        else:
            val_epoch_times.append(0)
            val_inference_times.append(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{config.model_name} - Métricas de Tempo - Run {run_id+1}', 
                 fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_epoch_times) + 1)
    
    # Tempo por época
    axes[0, 0].plot(epochs, train_epoch_times, label='Train', marker='o')
    axes[0, 0].plot(epochs, val_epoch_times, label='Validation', marker='s')
    if history['early_stop']['patience_used']:
        axes[0, 0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Tempo (s)')
    axes[0, 0].set_title('Tempo por Época')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Tempo médio por batch/inferência
    axes[0, 1].plot(epochs, train_batch_times, label='Train (Batch)', marker='o')
    axes[0, 1].plot(epochs, val_inference_times, label='Validation (Inference)', marker='s')
    if history['early_stop']['patience_used']:
        axes[0, 1].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Tempo (s)')
    axes[0, 1].set_title('Tempo Médio por Batch/Inferência')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Throughput
    train_throughput = [1/t if t > 0 else 0 for t in train_batch_times]
    val_throughput = [1/t if t > 0 else 0 for t in val_inference_times]
    axes[1, 0].plot(epochs, train_throughput, label='Train', marker='o')
    axes[1, 0].plot(epochs, val_throughput, label='Validation', marker='s')
    if history['early_stop']['patience_used']:
        axes[1, 0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Batches/Inferences por segundo')
    axes[1, 0].set_title('Throughput')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Tempo acumulado
    cumulative_train_time = np.cumsum(train_epoch_times)
    cumulative_val_time = np.cumsum(val_epoch_times)
    axes[1, 1].plot(epochs, cumulative_train_time, label='Train (Acumulado)', marker='o')
    axes[1, 1].plot(epochs, cumulative_val_time, label='Validation (Acumulado)', marker='s')
    if history['early_stop']['patience_used']:
        axes[1, 1].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Tempo Acumulado (s)')
    axes[1, 1].set_title('Tempo Acumulado de Execução')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'time_metrics_plots.png'), dpi=300)
    plt.close()

# ============================================
# RELATÓRIOS CONSOLIDADOS
# ============================================
def compute_average_results(config):
    all_final = []
    all_best = []
    for run_id in range(config.n_runs):
        run_dir = os.path.join(experiment_dir, f'run_{run_id}')
        json_path = os.path.join(run_dir, 'final_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                all_final.append(metrics)
                csv_path = os.path.join(run_dir, 'training_results.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    best_idx = df['val_dice'].idxmax()
                    all_best.append({
                        'run_id': int(run_id),
                        'best_epoch': int(df.loc[best_idx, 'epoch']),
                        'best_val_dice': float(df.loc[best_idx, 'val_dice']),
                        'best_val_loss': float(df.loc[best_idx, 'val_loss']),
                        'best_val_iou': float(df.loc[best_idx, 'val_iou']),
                        'best_val_accuracy': float(df.loc[best_idx, 'val_accuracy']),
                        'best_train_dice': float(df.loc[best_idx, 'train_dice']),
                    })

    if not all_final:
        print("⚠️ Nenhum resultado para médias!")
        return None

    df_final = pd.DataFrame(all_final)
    df_best = pd.DataFrame(all_best)

    stats = {}
    for col in df_final.columns:
        if col not in ['run_id', 'model_name'] and pd.api.types.is_numeric_dtype(df_final[col]):
            stats[col] = {
                'mean': float(df_final[col].mean()),
                'std': float(df_final[col].std()),
                'min': float(df_final[col].min()),
                'max': float(df_final[col].max()),
                'median': float(df_final[col].median())
            }

    df_final.to_csv(os.path.join(experiment_dir, 'summary_results.csv'), index=False)
    df_best.to_csv(os.path.join(experiment_dir, 'best_epochs_results.csv'), index=False)
    with open(os.path.join(experiment_dir, 'statistics.json'), 'w') as f:
        json.dump(convert_to_serializable(stats), f, indent=4)

    create_summary_report(df_final, df_best, stats, config)
    return df_final, df_best, stats

def create_summary_report(df_final, df_best, stats, config):
    report_path = os.path.join(reports_dir, f'RELATORIO_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"{' ' * 30}RELATÓRIO DE EXECUÇÕES\n")
        f.write("="*100 + "\n\n")
        f.write(f"📌 MODELO: {config.model_name}\n")
        f.write(f"📅 DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📐 TAMANHO DA IMAGEM: {config.img_size}x{config.img_size}\n")
        f.write(f"📊 EXECUÇÕES: {config.n_runs}\n")
        f.write(f"🔄 ÉPOCAS MÁXIMAS: {config.epochs}\n")
        f.write(f"⏳ PACIÊNCIA: {config.patience}\n")
        f.write(f"💻 DISPOSITIVO: {config.device}\n")
        f.write(f"⏱️ MEDIÇÃO DE TEMPO: {'Ativada' if config.measure_time else 'Desativada'}\n")
        f.write("-"*100 + "\n\n")

        f.write("📊 MÉTRICAS FINAIS (Média ± Desvio Padrão):\n")
        f.write("-"*50 + "\n")
        for metric, values in stats.items():
            if metric.startswith('final_val_') or metric.startswith('final_train_'):
                name = metric.replace('final_', '').replace('_', ' ').title()
                f.write(f"  {name}: {values['mean']:.4f} ± {values['std']:.4f} "
                        f"[{values['min']:.4f} - {values['max']:.4f}]\n")
        
        # Métricas de tempo
        f.write("\n" + "="*50 + "\n")
        f.write("⏱️ MÉTRICAS DE TEMPO (Média ± Desvio Padrão):\n")
        f.write("-"*50 + "\n")
        
        time_metrics = [
            'total_training_time', 
            'avg_train_epoch_time', 
            'avg_val_epoch_time',
            'final_train_epoch_total_time',
            'final_val_epoch_total_time',
            'final_train_batch_forward_time_mean',
            'final_val_inference_time_mean'
        ]
        
        time_metrics_found = False
        for metric in time_metrics:
            if metric in stats:
                metric_name = metric.replace('_', ' ').title()
                f.write(f"  {metric_name}: {stats[metric]['mean']:.2f} ± {stats[metric]['std']:.2f} "
                       f"[{stats[metric]['min']:.2f} - {stats[metric]['max']:.2f}] (s)\n")
                time_metrics_found = True
        
        if not time_metrics_found:
            f.write("  ⚠️ Nenhuma métrica de tempo disponível\n")

        f.write("\n🏆 MELHORES POR EXECUÇÃO:\n")
        f.write("-"*50 + "\n")
        for _, row in df_best.iterrows():
            f.write(f"  Run {int(row['run_id'])}: Época {int(row['best_epoch'])} - "
                    f"Dice: {row['best_val_dice']:.4f} - IoU: {row['best_val_iou']:.4f}\n")

        f.write("\n📈 MÉDIAS DOS MELHORES:\n")
        f.write("-"*50 + "\n")
        for col in df_best.columns:
            if col != 'run_id' and pd.api.types.is_numeric_dtype(df_best[col]):
                name = col.replace('best_', '').replace('_', ' ').title()
                f.write(f"  {name}: {df_best[col].mean():.4f} ± {df_best[col].std():.4f}\n")

        f.write("\n📋 DETALHES POR EXECUÇÃO:\n")
        f.write("-"*50 + "\n")
        for _, row in df_final.iterrows():
            f.write(f"\n  Run {int(row['run_id'])}:\n")
            f.write(f"    Melhor Dice: {row['best_val_dice']:.4f} (época {int(row['best_epoch'])})\n")
            f.write(f"    Dice Final: {row['final_val_dice']:.4f}\n")
            f.write(f"    IoU Final: {row['final_val_iou']:.4f}\n")
            f.write(f"    Accuracy Final: {row['final_val_accuracy']:.4f}\n")
            f.write(f"    Sensitivity Final: {row['final_val_sensitivity']:.4f}\n")
            f.write(f"    Specificity Final: {row['final_val_specificity']:.4f}\n")
            
            if 'total_training_time' in row:
                f.write(f"    ⏱️ Tempo total de treinamento: {row['total_training_time']:.2f}s\n")
            if 'avg_train_epoch_time' in row:
                f.write(f"    ⏱️ Tempo médio por época (train): {row['avg_train_epoch_time']:.2f}s\n")
            if 'avg_val_epoch_time' in row:
                f.write(f"    ⏱️ Tempo médio por época (val): {row['avg_val_epoch_time']:.2f}s\n")
            
            if 'early_stopped' in row:
                f.write(f"    Early Stopping: {'Sim' if row['early_stopped'] else 'Não'}\n")
                f.write(f"    Épocas treinadas: {int(row['total_epochs'])}\n")

        f.write("\n" + "="*100 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("="*100 + "\n")
    print(f"✅ Relatório consolidado: {report_path}")

# ============================================
# TESTE COM MEDIÇÃO DE TEMPO
# ============================================
def test_model(model, test_loader, device, run_id, config):
    model.eval()
    all_metrics = []
    
    total_start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f'Test Run {run_id+1}'):
            if config.measure_time:
                start_time = time.time()
            
            images = images.to(device)
            outputs = model(images)
            
            if config.measure_time:
                inference_times.append(time.time() - start_time)
            
            metrics = compute_metrics(outputs.cpu().numpy(), masks.cpu().numpy())
            all_metrics.append(metrics)

    total_time = time.time() - total_start_time
    
    final_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    
    if config.measure_time and inference_times:
        final_metrics['test_total_time'] = float(total_time)
        final_metrics['test_inference_time_mean'] = float(np.mean(inference_times))
        final_metrics['test_inference_time_std'] = float(np.std(inference_times))
        final_metrics['test_inference_time_min'] = float(np.min(inference_times))
        final_metrics['test_inference_time_max'] = float(np.max(inference_times))
        final_metrics['test_inferences_per_second'] = float(len(inference_times) / total_time)
        final_metrics['test_images_per_second'] = float(len(test_loader.dataset) / total_time)
    
    if config.save_results:
        run_dir = os.path.join(experiment_dir, f'run_{run_id}')
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame(all_metrics).to_csv(os.path.join(run_dir, 'test_results.csv'), index=False)
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
            json.dump(convert_to_serializable(final_metrics), f, indent=4)
    return final_metrics

def test_model_average(config):
    test_dataset = FundusSegmentationDataset(
        config.test_images_dir, config.test_masks_dir, img_size=config.img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    all_test_metrics = []
    for run_id in range(config.n_runs):
        print(f"\n📈 Testando Run {run_id+1}/{config.n_runs}")
        model = SwinUNet(num_classes=1)
        model_path = f"{config.best_model_path}_run{run_id}"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model = model.to(config.device)
            metrics = test_model(model, test_loader, config.device, run_id, config)
            all_test_metrics.append(metrics)
        else:
            print(f"⚠️ Modelo run {run_id} não encontrado!")

    if all_test_metrics:
        test_df = pd.DataFrame(all_test_metrics)
        test_df.to_csv(os.path.join(experiment_dir, 'test_summary.csv'), index=False)
        test_stats = {}
        for col in test_df.columns:
            test_stats[col] = {
                'mean': float(test_df[col].mean()),
                'std': float(test_df[col].std()),
                'min': float(test_df[col].min()),
                'max': float(test_df[col].max())
            }
        with open(os.path.join(experiment_dir, 'test_statistics.json'), 'w') as f:
            json.dump(test_stats, f, indent=4)
        print("\n" + "="*50)
        print(f"RESULTADOS DO TESTE - {config.model_name} (Média entre runs)")
        print("="*50)
        for metric, values in test_stats.items():
            if 'time' in metric.lower():
                print(f"⏱️  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
            else:
                print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        return test_df, test_stats
    return None, None

# ============================================
# MAIN
# ============================================
def main():
    print("="*70)
    print(f"{' ' * 20}🚀 {config.model_name}")
    print(f"{' ' * 15}Segmentação de Vasos em Fundoscopia")
    print("="*70)

    print(f"\n📊 CONFIGURAÇÕES:")
    print(f"  Device: {config.device}")
    print(f"  Imagem: {config.img_size}x{config.img_size}")
    print(f"  Execuções: {config.n_runs}")
    print(f"  Épocas: {config.epochs}")
    print(f"  Paciência: {config.patience}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  LR: {config.learning_rate}")
    print(f"  Medição de tempo: {'Ativada' if config.measure_time else 'Desativada'}")
    print(f"  Diretório: {experiment_dir}")

    print("\n📂 Carregando dados...")
    try:
        full_dataset = FundusSegmentationDataset(
            config.train_images_dir, config.train_masks_dir, img_size=config.img_size
        )
    except Exception as e:
        print(f"❌ Erro: {e}")
        return

    if len(full_dataset) < 10:
        print(f"❌ Poucos dados: {len(full_dataset)}")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if train_size < 2 or val_size < 2:
        print(f"❌ Dados insuficientes: Treino={train_size}, Val={val_size}")
        return

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)

    print(f"  Treino: {len(train_dataset)} imagens")
    print(f"  Validação: {len(val_dataset)} imagens")

    all_histories = []
    all_best_dices = []
    all_training_times = []

    for run_id in range(config.n_runs):
        run_start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"# EXECUÇÃO {run_id+1}/{config.n_runs} - {config.model_name}")
        print(f"{'#'*70}")

        try:
            model = SwinUNet(num_classes=config.num_classes, pretrained=config.pretrained)
            model = model.to(config.device)
            test_input = torch.randn(1, 3, 224, 224).to(config.device)
            test_output = model(test_input)
            print(f"✅ Forward pass OK - Input: {test_input.shape}, Output: {test_output.shape}")
        except Exception as e:
            print(f"❌ Erro ao criar modelo: {e}")
            import traceback
            traceback.print_exc()
            continue

        history, best_dice = train_model(model, train_loader, val_loader, config, run_id)
        all_histories.append(history)
        all_best_dices.append(best_dice)
        
        run_total_time = time.time() - run_start_time
        all_training_times.append(run_total_time)
        print(f"\n⏱️ Tempo total da execução {run_id+1}: {run_total_time:.2f}s")

        if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
            try:
                test_dataset = FundusSegmentationDataset(
                    config.test_images_dir, config.test_masks_dir, img_size=config.img_size
                )
                test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
                model_path = f"{config.best_model_path}_run{run_id}"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=config.device))
                    model = model.to(config.device)
                    test_model(model, test_loader, config.device, run_id, config)
            except Exception as e:
                print(f"⚠️ Erro no teste: {e}")

        del model
        torch.cuda.empty_cache()

    if all_best_dices:
        print("\n" + "="*70)
        print("📊 CALCULANDO MÉDIAS ENTRE EXECUÇÕES")
        print("="*70)
        best_dice_array = np.array(all_best_dices)
        print("\nMelhor Dice por execução:")
        for i, d in enumerate(all_best_dices):
            print(f"  Run {i+1}: {d:.4f}")
        print(f"\nMédia: {best_dice_array.mean():.4f} ± {best_dice_array.std():.4f}")
        
        if all_training_times:
            print(f"\n⏱️ Tempo total por execução:")
            for i, t in enumerate(all_training_times):
                print(f"  Run {i+1}: {t:.2f}s")
            print(f"\n⏱️ Tempo médio por execução: {np.mean(all_training_times):.2f}s ± {np.std(all_training_times):.2f}s")

        compute_average_results(config)

        if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
            test_df, test_stats = test_model_average(config)

    print("\n" + "="*70)
    print(f"✅ EXPERIMENTO CONCLUÍDO - {config.model_name}")
    print("="*70)
    print(f"\n📁 Resultados: {experiment_dir}")
    print(f"📁 Relatórios: {reports_dir}")
    
    if all_training_times:
        print(f"\n⏱️ RESUMO DE TEMPO:")
        print(f"  Tempo total médio por execução: {np.mean(all_training_times):.2f}s")
        print(f"  Tempo total de todas as execuções: {np.sum(all_training_times):.2f}s")

if __name__ == "__main__":
    main()