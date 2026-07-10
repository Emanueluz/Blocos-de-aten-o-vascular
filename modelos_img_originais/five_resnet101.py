import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import pandas as pd
from datetime import datetime
import json
import warnings
import time  # ADICIONADO

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Dados
    train_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original'
    train_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth'
    test_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Original'
    test_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth'
    
    img_size = 224
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    
    # NOVAS CONFIGURAÇÕES
    n_runs = 5  # Número de execuções para média dos resultados
    save_results = True
    results_dir = './results'
    model_name = 'ResNet101_UNet'  # Nome da rede para os relatórios
    experiment_name = f'{model_name}_{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}'
    
    # CONFIGURAÇÕES DE EARLY STOPPING (PACIÊNCIA)
    patience = 10  # Número de épocas sem melhora para parar
    min_delta = 0.001  # Melhora mínima considerada significativa
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = './best_resnet101_segmentation.pth'
    
    # NOVA CONFIGURAÇÃO: Métricas de tempo
    measure_time = True  # Ativar/desativar medição de tempo

config = Config()

# Criar diretório de resultados principal
os.makedirs(config.results_dir, exist_ok=True)

# Criar diretório específico para esta rede
network_dir = os.path.join(config.results_dir, config.model_name)
os.makedirs(network_dir, exist_ok=True)

# Criar diretório do experimento dentro da pasta da rede
experiment_dir = os.path.join(network_dir, config.experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

# Criar diretório para relatórios consolidados
reports_dir = os.path.join(network_dir, f'RELATORIO_DAS_EXECUCOES')
os.makedirs(reports_dir, exist_ok=True)

print(f"✅ Diretórios criados:")
print(f"   Network: {network_dir}")
print(f"   Experimento: {experiment_dir}")
print(f"   Relatórios: {reports_dir}")
print(f"   Device: {config.device}")
print()

# ============================================
# FUNÇÃO AUXILIAR PARA CONVERTER PARA TIPOS PYTHON
# ============================================
def convert_to_serializable(obj):
    """Converte objetos numpy/pandas para tipos serializáveis em JSON"""
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

# ============================================
# MODELO (mantido igual)
# ============================================
class ResNet101UNet(nn.Module):
    """
    U-Net com encoder ResNet101 - VERSÃO FINAL FUNCIONAL
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNet101UNet, self).__init__()
        
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        self.center = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(256 + 1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e1_pool = self.maxpool(e1)
        
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        c = self.center(e5)
        
        d5 = self.up5(c)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        out = self.sigmoid(out)
        
        return out

# ============================================
# DATASET (mantido igual)
# ============================================
class FundusSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None, img_size=224):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*.*g')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*.*g')))
        
        assert len(self.images_paths) == len(self.masks_paths), "Número de imagens e máscaras diferente!"
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask

# ============================================
# LOSS E MÉTRICAS
# ============================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()
    
    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        return bce_loss + dice_loss

def compute_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).astype(np.uint8)
    targets_binary = targets.astype(np.uint8)
    
    acc = accuracy_score(targets_binary.flatten(), preds_binary.flatten())
    
    intersection = (preds_binary & targets_binary).sum()
    dice = (2.0 * intersection) / (preds_binary.sum() + targets_binary.sum() + 1e-6)
    
    union = (preds_binary | targets_binary).sum()
    iou = intersection / (union + 1e-6)
    
    sens = intersection / (targets_binary.sum() + 1e-6)
    
    tn = ((1 - preds_binary) & (1 - targets_binary)).sum()
    spec = tn / ((1 - targets_binary).sum() + 1e-6)
    
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
def train_epoch(model, train_loader, criterion, optimizer, device, measure_time=True):
    model.train()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    epoch_start_time = time.time()
    batch_times = []
    data_load_times = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(progress_bar):
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
        
        batch_metrics = compute_metrics(
            outputs.detach().cpu().numpy(),
            masks.detach().cpu().numpy()
        )
        for k in metrics:
            metrics[k] += batch_metrics[k]
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    num_batches = len(train_loader)
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

def validate_epoch(model, val_loader, criterion, device, measure_time=True):
    model.eval()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    epoch_start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            if measure_time:
                start_time = time.time()
            
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            if measure_time:
                inference_times.append(time.time() - start_time)
            
            running_loss += loss.item()
            
            batch_metrics = compute_metrics(
                outputs.cpu().numpy(),
                masks.cpu().numpy()
            )
            for k in metrics:
                metrics[k] += batch_metrics[k]
    
    num_batches = len(val_loader)
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
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
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
    
    print(f"\n🚀 Treinamento Run {run_id+1}/{config.n_runs} - {config.model_name}")
    print(f"Dispositivo: {config.device}")
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Paciência: {config.patience} épocas")
    print(f"Épocas máximas: {config.epochs}")
    print(f"⏱️ Medição de tempo: {'Ativada' if config.measure_time else 'Desativada'}\n")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Run {run_id+1} - Época {epoch+1}/{config.epochs}")
        
        train_loss, train_metrics, train_time = train_epoch(
            model, train_loader, criterion, optimizer, config.device, config.measure_time
        )
        val_loss, val_metrics, val_time = validate_epoch(
            model, val_loader, criterion, config.device, config.measure_time
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)
        
        print(f"\n📊 Treino - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"📊 Validação - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        if config.measure_time:
            print(f"⏱️  Treino: {train_time.get('epoch_total_time', 0):.2f}s | "
                  f"Batch: {train_time.get('batch_forward_time_mean', 0):.3f}s")
            print(f"⏱️  Validação: {val_time.get('epoch_total_time', 0):.2f}s | "
                  f"Inferência: {val_time.get('inference_time_mean', 0):.3f}s")
        
        # Verificar melhora
        if val_metrics['dice'] > best_dice + config.min_delta:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"{config.best_model_path}_run{run_id}")
            print(f"✅ Melhor modelo salvo! Dice: {best_dice:.4f} (época {epoch+1})")
        else:
            patience_counter += 1
            print(f"⏳ Paciência: {patience_counter}/{config.patience} (melhor Dice: {best_dice:.4f} na época {best_epoch+1})")
        
        # Early Stopping
        if patience_counter >= config.patience:
            stopped_epoch = epoch + 1
            print(f"\n🛑 Early stopping ativado! Parando treinamento na época {stopped_epoch}")
            print(f"Melhor Dice: {best_dice:.4f} (época {best_epoch+1})")
            break
    
    total_training_time = time.time() - training_start_time
    
    # Salvar informações de early stopping
    history['early_stop'] = {
        'stopped_epoch': stopped_epoch,
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'patience_used': patience_counter >= config.patience,
        'total_training_time': total_training_time
    }
    
    # Salvar histórico desta execução
    if config.save_results:
        save_run_results(history, run_id, config)
    
    return history, best_dice

# ============================================
# FUNÇÕES PARA SALVAR RESULTADOS
# ============================================
def save_run_results(history, run_id, config):
    """Salva os resultados de uma execução em CSV"""
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Criar DataFrame com todas as métricas por época
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
    
    # Adicionar métricas detalhadas
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        data[f'train_{metric}'] = [m[metric] for m in history['train_metrics']]
        data[f'val_{metric}'] = [m[metric] for m in history['val_metrics']]
    
    df = pd.DataFrame(data)
    
    # Salvar CSV
    csv_path = os.path.join(run_dir, 'training_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Resultados salvos em: {csv_path}")
    
    # Salvar métricas finais
    final_metrics = {
        'run_id': int(run_id),
        'model_name': config.model_name,
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
    
    # Adicionar últimas métricas
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        final_metrics[f'final_val_{metric}'] = float(history['val_metrics'][-1][metric])
        final_metrics[f'final_train_{metric}'] = float(history['train_metrics'][-1][metric])
    
    # Converter para tipos serializáveis
    final_metrics = convert_to_serializable(final_metrics)
    
    # Salvar métricas finais em JSON
    json_path = os.path.join(run_dir, 'final_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Plotar e salvar gráficos
    plot_training_history(history, run_id, config)
    plot_time_metrics(history, run_id, config)
    
    return df, final_metrics

def plot_training_history(history, run_id, config):
    """Plota e salva os gráficos de treinamento"""
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Título principal
    fig.suptitle(f'{config.model_name} - Run {run_id+1}', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    if 'early_stop' in history and history['early_stop']['patience_used']:
        axes[0, 0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice
    axes[0, 1].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[0, 1].plot(history['val_dice'], label='Val Dice', marker='s')
    if 'early_stop' in history and history['early_stop']['patience_used']:
        axes[0, 1].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Coefficient')
    axes[0, 1].set_title('Dice Score Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    train_iou = [m['iou'] for m in history['train_metrics']]
    val_iou = [m['iou'] for m in history['val_metrics']]
    axes[1, 0].plot(train_iou, label='Train IoU', marker='o')
    axes[1, 0].plot(val_iou, label='Val IoU', marker='s')
    if 'early_stop' in history and history['early_stop']['patience_used']:
        axes[1, 0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('IoU Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Accuracy
    train_acc = [m['accuracy'] for m in history['train_metrics']]
    val_acc = [m['accuracy'] for m in history['val_metrics']]
    axes[1, 1].plot(train_acc, label='Train Accuracy', marker='o')
    axes[1, 1].plot(val_acc, label='Val Accuracy', marker='s')
    if 'early_stop' in history and history['early_stop']['patience_used']:
        axes[1, 1].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
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
    if 'early_stop' in history and history['early_stop']['patience_used']:
        axes[0, 0].axvline(x=history['early_stop']['stopped_epoch']-1, color='r', linestyle='--', label='Early Stop')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Tempo (s)')
    axes[0, 0].set_title('Tempo por Época')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Tempo médio por batch/inferência
    axes[0, 1].plot(epochs, train_batch_times, label='Train (Batch)', marker='o')
    axes[0, 1].plot(epochs, val_inference_times, label='Validation (Inference)', marker='s')
    if 'early_stop' in history and history['early_stop']['patience_used']:
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
    if 'early_stop' in history and history['early_stop']['patience_used']:
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
    if 'early_stop' in history and history['early_stop']['patience_used']:
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
# FUNÇÃO PARA CALCULAR MÉDIAS ENTRE RUNS
# ============================================
def compute_average_results(config):
    """Calcula a média dos resultados entre todas as execuções"""
    all_final_metrics = []
    all_best_metrics = []
    
    # Coletar todos os resultados
    for run_id in range(config.n_runs):
        run_dir = os.path.join(experiment_dir, f'run_{run_id}')
        json_path = os.path.join(run_dir, 'final_metrics.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                all_final_metrics.append(metrics)
                
                # Coletar melhores métricas por época
                csv_path = os.path.join(run_dir, 'training_results.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    best_idx = df['val_dice'].idxmax()
                    best_metrics = {
                        'run_id': int(run_id),
                        'best_epoch': int(df.loc[best_idx, 'epoch']),
                        'best_val_dice': float(df.loc[best_idx, 'val_dice']),
                        'best_val_loss': float(df.loc[best_idx, 'val_loss']),
                        'best_val_iou': float(df.loc[best_idx, 'val_iou']),
                        'best_val_accuracy': float(df.loc[best_idx, 'val_accuracy']),
                        'best_train_dice': float(df.loc[best_idx, 'train_dice']),
                    }
                    all_best_metrics.append(best_metrics)
    
    if not all_final_metrics:
        print("⚠️ Nenhum resultado encontrado para calcular médias!")
        return None
    
    # Calcular médias
    df_final = pd.DataFrame(all_final_metrics)
    df_best = pd.DataFrame(all_best_metrics)
    
    # Estatísticas descritivas
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
    
    # Converter para serializável
    stats = convert_to_serializable(stats)
    
    # Salvar resultados consolidados
    summary_path = os.path.join(experiment_dir, 'summary_results.csv')
    df_final.to_csv(summary_path, index=False)
    
    # Salvar melhores resultados
    best_path = os.path.join(experiment_dir, 'best_epochs_results.csv')
    df_best.to_csv(best_path, index=False)
    
    # Salvar estatísticas
    stats_path = os.path.join(experiment_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Criar relatório consolidado
    create_summary_report(df_final, df_best, stats, config)
    
    return df_final, df_best, stats

def create_summary_report(df_final, df_best, stats, config):
    """Cria um relatório consolidado em texto com formatação melhorada"""
    report_path = os.path.join(reports_dir, f'RELATORIO_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"{' ' * 30}RELATÓRIO DE EXECUÇÕES\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"📌 MODELO: {config.model_name}\n")
        f.write(f"📅 DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📊 NÚMERO DE EXECUÇÕES: {config.n_runs}\n")
        f.write(f"🔄 ÉPOCAS MÁXIMAS: {config.epochs}\n")
        f.write(f"⏳ PACIÊNCIA (EARLY STOPPING): {config.patience} épocas\n")
        f.write(f"📈 MIN_DELTA: {config.min_delta}\n")
        f.write(f"💻 DISPOSITIVO: {config.device}\n")
        f.write(f"⏱️ MEDIÇÃO DE TEMPO: {'Ativada' if config.measure_time else 'Desativada'}\n")
        f.write(f"📁 EXPERIMENTO: {config.experiment_name}\n")
        f.write("-"*100 + "\n\n")
        
        # Métricas finais
        f.write("📊 MÉTRICAS FINAIS (Média ± Desvio Padrão):\n")
        f.write("-"*50 + "\n")
        for metric, values in stats.items():
            if metric.startswith('final_val_') or metric.startswith('final_train_'):
                metric_name = metric.replace('final_', '').replace('_', ' ').title()
                f.write(f"  {metric_name}: {values['mean']:.4f} ± {values['std']:.4f} "
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
        
        f.write("\n" + "-"*50 + "\n")
        f.write("🏆 MELHORES RESULTADOS POR EXECUÇÃO:\n")
        f.write("-"*50 + "\n")
        for _, row in df_best.iterrows():
            f.write(f"  Run {int(row['run_id'])}: Época {int(row['best_epoch'])} - "
                   f"Dice: {row['best_val_dice']:.4f} - IoU: {row['best_val_iou']:.4f}\n")
        
        f.write("\n" + "-"*50 + "\n")
        f.write("📈 MÉDIAS DOS MELHORES RESULTADOS:\n")
        f.write("-"*50 + "\n")
        for col in df_best.columns:
            if col != 'run_id' and pd.api.types.is_numeric_dtype(df_best[col]):
                col_name = col.replace('best_', '').replace('_', ' ').title()
                f.write(f"  {col_name}: {df_best[col].mean():.4f} ± {df_best[col].std():.4f}\n")
        
        # Resumo das execuções
        f.write("\n" + "-"*50 + "\n")
        f.write("📋 DETALHES POR EXECUÇÃO:\n")
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
        f.write(f"{' ' * 40}FIM DO RELATÓRIO\n")
        f.write("="*100 + "\n")
    
    print(f"✅ Relatório consolidado salvo em: {report_path}")

# ============================================
# FUNÇÃO DE TESTE COM MEDIÇÃO DE TEMPO
# ============================================
def test_model(model, test_loader, device, run_id, config):
    """Testa o modelo e salva os resultados com medição de tempo"""
    model.eval()
    all_metrics = []
    predictions = []
    ground_truths = []
    
    total_start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f'Testing Run {run_id+1}'):
            if config.measure_time:
                start_time = time.time()
            
            images = images.to(device)
            outputs = model(images)
            
            if config.measure_time:
                inference_times.append(time.time() - start_time)
            
            metrics = compute_metrics(outputs.cpu().numpy(), masks.numpy())
            all_metrics.append(metrics)
            
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(masks.numpy())
    
    total_time = time.time() - total_start_time
    
    # Calcular médias
    final_metrics = {}
    for k in all_metrics[0].keys():
        final_metrics[k] = np.mean([m[k] for m in all_metrics])
    
    # Adicionar métricas de tempo
    if config.measure_time and inference_times:
        final_metrics['test_total_time'] = float(total_time)
        final_metrics['test_inference_time_mean'] = float(np.mean(inference_times))
        final_metrics['test_inference_time_std'] = float(np.std(inference_times))
        final_metrics['test_inference_time_min'] = float(np.min(inference_times))
        final_metrics['test_inference_time_max'] = float(np.max(inference_times))
        final_metrics['test_inferences_per_second'] = float(len(inference_times) / total_time)
        final_metrics['test_images_per_second'] = float(len(test_loader.dataset) / total_time)
    
    # Converter para float
    final_metrics = {k: float(v) for k, v in final_metrics.items()}
    
    # Salvar resultados do teste
    if config.save_results:
        run_dir = os.path.join(experiment_dir, f'run_{run_id}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Salvar métricas do teste
        test_df = pd.DataFrame(all_metrics)
        test_df.to_csv(os.path.join(run_dir, 'test_results.csv'), index=False)
        
        # Salvar métricas finais
        test_metrics_path = os.path.join(run_dir, 'test_metrics.json')
        with open(test_metrics_path, 'w') as f:
            json.dump(convert_to_serializable(final_metrics), f, indent=4)
    
    return final_metrics

def test_model_average(config):
    """Testa todos os modelos treinados e calcula a média"""
    test_dataset = FundusSegmentationDataset(
        config.test_images_dir,
        config.test_masks_dir,
        img_size=config.img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    all_test_metrics = []
    
    for run_id in range(config.n_runs):
        print(f"\n📈 Testando Run {run_id+1}/{config.n_runs}")
        
        # Carregar modelo
        model = ResNet101UNet(num_classes=1)
        model_path = f"{config.best_model_path}_run{run_id}"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model = model.to(config.device)
            
            metrics = test_model(model, test_loader, config.device, run_id, config)
            all_test_metrics.append(metrics)
        else:
            print(f"⚠️ Modelo da run {run_id} não encontrado!")
    
    if all_test_metrics:
        # Calcular médias
        test_df = pd.DataFrame(all_test_metrics)
        
        # Salvar resultados consolidados do teste
        test_summary_path = os.path.join(experiment_dir, 'test_summary.csv')
        test_df.to_csv(test_summary_path, index=False)
        
        # Calcular estatísticas
        test_stats = {}
        for col in test_df.columns:
            test_stats[col] = {
                'mean': float(test_df[col].mean()),
                'std': float(test_df[col].std()),
                'min': float(test_df[col].min()),
                'max': float(test_df[col].max())
            }
        
        # Converter para serializável
        test_stats = convert_to_serializable(test_stats)
        
        # Salvar estatísticas do teste
        test_stats_path = os.path.join(experiment_dir, 'test_statistics.json')
        with open(test_stats_path, 'w') as f:
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
# FUNÇÃO PRINCIPAL MODIFICADA
# ============================================
def main():
    print("="*70)
    print(f"{' ' * 20}🚀 {config.model_name}")
    print(f"{' ' * 15}Segmentação de Vasos em Fundoscopia")
    print("="*70)
    
    print(f"\n📊 CONFIGURAÇÕES:")
    print(f"  Device: {config.device}")
    print(f"  Execuções: {config.n_runs}")
    print(f"  Épocas máximas: {config.epochs}")
    print(f"  Paciência: {config.patience} épocas")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Medição de tempo: {'Ativada' if config.measure_time else 'Desativada'}")
    print(f"  Diretório: {experiment_dir}")
    
    # Preparar dados
    print("\n📂 Carregando dados...")
    full_dataset = FundusSegmentationDataset(
        config.train_images_dir, 
        config.train_masks_dir,
        img_size=config.img_size
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    print(f"  Treino: {len(train_dataset)} imagens")
    print(f"  Validação: {len(val_dataset)} imagens")
    
    # Executar múltiplos treinamentos
    all_histories = []
    all_best_dices = []
    all_training_times = []
    
    for run_id in range(config.n_runs):
        run_start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"# EXECUÇÃO {run_id+1}/{config.n_runs} - {config.model_name}")
        print(f"{'#'*70}")
        
        # Criar modelo
        model = ResNet101UNet(num_classes=1, pretrained=True)
        model = model.to(config.device)
        
        # Treinar
        history, best_dice = train_model(model, train_loader, val_loader, config, run_id)
        all_histories.append(history)
        all_best_dices.append(best_dice)
        
        run_total_time = time.time() - run_start_time
        all_training_times.append(run_total_time)
        print(f"\n⏱️ Tempo total da execução {run_id+1}: {run_total_time:.2f}s")
        
        # Testar se houver dados de teste
        if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
            test_dataset = FundusSegmentationDataset(
                config.test_images_dir,
                config.test_masks_dir,
                img_size=config.img_size
            )
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
            
            model_path = f"{config.best_model_path}_run{run_id}"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=config.device))
                model = model.to(config.device)
                test_model(model, test_loader, config.device, run_id, config)
        
        # Liberar memória
        del model
        torch.cuda.empty_cache()
    
    # Calcular médias entre runs
    print("\n" + "="*70)
    print("📊 CALCULANDO MÉDIAS ENTRE EXECUÇÕES")
    print("="*70)
    
    # Calcular estatísticas dos melhores dices
    best_dice_array = np.array(all_best_dices)
    print(f"\nMelhor Dice por execução:")
    for i, dice in enumerate(all_best_dices):
        print(f"  Run {i+1}: {dice:.4f}")
    print(f"\nMédia dos melhores Dices: {best_dice_array.mean():.4f} ± {best_dice_array.std():.4f}")
    
    if all_training_times:
        print(f"\n⏱️ Tempo total por execução:")
        for i, t in enumerate(all_training_times):
            print(f"  Run {i+1}: {t:.2f}s")
        print(f"\n⏱️ Tempo médio por execução: {np.mean(all_training_times):.2f}s ± {np.std(all_training_times):.2f}s")
    
    # Salvar resultados consolidados
    compute_average_results(config)
    
    # Testar todos os modelos e calcular média
    if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
        test_df, test_stats = test_model_average(config)
    
    print("\n" + "="*70)
    print(f"✅ EXPERIMENTO CONCLUÍDO - {config.model_name}")
    print("="*70)
    print(f"\n📁 Resultados salvos em: {experiment_dir}")
    print(f"📁 Relatórios consolidados em: {reports_dir}")
    
    if all_training_times:
        print(f"\n⏱️ RESUMO DE TEMPO:")
        print(f"  Tempo total médio por execução: {np.mean(all_training_times):.2f}s")
        print(f"  Tempo total de todas as execuções: {np.sum(all_training_times):.2f}s")

# ============================================
# FUNÇÃO PARA VISUALIZAR RESULTADOS
# ============================================
def visualize_results(config):
    """Visualiza os resultados consolidados do experimento"""
    # Carregar resumo
    summary_path = os.path.join(experiment_dir, 'summary_results.csv')
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        
        # Plotar comparação entre runs
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{config.model_name} - Comparação entre Execuções', fontsize=16, fontweight='bold')
        
        # Dice
        axes[0, 0].bar(range(len(df)), df['final_val_dice'])
        axes[0, 0].axhline(y=df['final_val_dice'].mean(), color='r', linestyle='--', 
                          label=f'Média: {df["final_val_dice"].mean():.4f}')
        axes[0, 0].set_xlabel('Run')
        axes[0, 0].set_ylabel('Dice')
        axes[0, 0].set_title('Dice Final por Execução')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU
        axes[0, 1].bar(range(len(df)), df['final_val_iou'])
        axes[0, 1].axhline(y=df['final_val_iou'].mean(), color='r', linestyle='--',
                          label=f'Média: {df["final_val_iou"].mean():.4f}')
        axes[0, 1].set_xlabel('Run')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('IoU Final por Execução')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].bar(range(len(df)), df['final_val_accuracy'])
        axes[1, 0].axhline(y=df['final_val_accuracy'].mean(), color='r', linestyle='--',
                          label=f'Média: {df["final_val_accuracy"].mean():.4f}')
        axes[1, 0].set_xlabel('Run')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy Final por Execução')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Sensitivity
        axes[1, 1].bar(range(len(df)), df['final_val_sensitivity'])
        axes[1, 1].axhline(y=df['final_val_sensitivity'].mean(), color='r', linestyle='--',
                          label=f'Média: {df["final_val_sensitivity"].mean():.4f}')
        axes[1, 1].set_xlabel('Run')
        axes[1, 1].set_ylabel('Sensitivity')
        axes[1, 1].set_title('Sensitivity Final por Execução')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'comparacao_execucoes_{config.model_name}.png'), dpi=300)
        plt.close()
        
        print("\n📊 Estatísticas das execuções:")
        print(df.describe())

if __name__ == "__main__":
    main()
    
    # Opcional: visualizar resultados após o experimento
    visualize_results(config)