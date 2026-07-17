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
import time

warnings.filterwarnings('ignore')

print("="*50)
print("DIAGNÓSTICO DE GPU")                                                                           
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("❌ CUDA NÃO está disponível!")
    print("Usando CPU para treinamento...")
print("="*50)

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Dados
    train_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/cinza'
    train_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth'
    test_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/test/cinza'
    test_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth'
        
    
    num_classes = 1
    img_size = 224
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    
    n_runs = 5
    save_results = True
    results_dir = './results_cinza'
    model_name = 'MobileNetV2_UNet_cinza'
    experiment_name = f'{model_name}_{datetime.now().strftime("%d_%m_%d_%H:%M:%S")}'
    
    patience = 5
    min_delta = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = './best_MobileNetV2UNetcinza_segmentation.pth'
    
    num_workers = 4
    pin_memory = True if torch.cuda.is_available() else False
    pretrained = True
    scheduler_patience = 5
    scheduler_factor = 0.5
    
    measure_time = True
    input_mode = 'grayscale'

config = Config()

# Criar diretórios
os.makedirs(config.results_dir, exist_ok=True)

network_dir = os.path.join(config.results_dir, config.model_name)
os.makedirs(network_dir, exist_ok=True)

experiment_dir = os.path.join(network_dir, config.experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

# Criar diretório para modelos salvos
models_dir = os.path.join(experiment_dir, 'saved_models')
os.makedirs(models_dir, exist_ok=True)

reports_dir = os.path.join(network_dir, f'RELATORIO_DAS_EXECUCOES')
os.makedirs(reports_dir, exist_ok=True)

# Criar diretório para resultados de teste
test_results_dir = os.path.join(experiment_dir, 'test_results')
os.makedirs(test_results_dir, exist_ok=True)

print(f"✅ Diretórios criados:")
print(f"   Network: {network_dir}")
print(f"   Experimento: {experiment_dir}")
print(f"   Modelos salvos: {models_dir}")
print(f"   Relatórios: {reports_dir}")
print(f"   Teste: {test_results_dir}")
print(f"   Device: {config.device}")
print(f"   Modo de entrada: {config.input_mode}")
print()

# ============================================
# FUNÇÃO AUXILIAR
# ============================================
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

# ============================================
# MODELO MobileNetV2UNet MODIFICADO PARA GRAYSCALE
# ============================================
class MobileNetV2UNet(nn.Module):
    """
    U-Net com encoder MobileNetV2 - MODIFICADO PARA GRAYSCALE
    """
    def __init__(self, num_classes=1, pretrained=True, input_channels=1):
        super(MobileNetV2UNet, self).__init__()
        
        self.input_channels = input_channels
        
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
        
        features = list(mobilenet.features)
        
        # CORREÇÃO: Acessar a camada de convolução dentro do Conv2dNormActivation
        first_conv_block = features[0]
        # Pegar a primeira camada do bloco (que é a convolução)
        if hasattr(first_conv_block, 'conv'):
            first_conv = first_conv_block.conv
        else:
            # Fallback para versões mais antigas
            first_conv = first_conv_block[0]
        
        # Criar nova convolução com o número correto de canais de entrada
        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        if pretrained:
            with torch.no_grad():
                original_weights = first_conv.weight
                # Média dos pesos dos canais RGB para criar um único canal
                new_weights = original_weights.mean(dim=1, keepdim=True)
                new_conv.weight.data = new_weights
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data
        else:
            new_conv.reset_parameters()
        
        # Substituir a convolução no bloco
        if hasattr(first_conv_block, 'conv'):
            first_conv_block.conv = new_conv
        else:
            first_conv_block[0] = new_conv
        
        features[0] = first_conv_block
        features = nn.Sequential(*features)
        
        self.enc1 = features[:1]
        self.enc2 = features[1:3]
        self.enc3 = features[3:6]
        self.enc4 = features[6:13]
        self.enc5 = features[13:18]
        
        self.center = nn.Sequential(
            nn.Conv2d(320, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128 + 96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(16 + 32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        center = self.center(e5)
        
        d5 = self.up5(center)
        if d5.shape[2:] != e4.shape[2:]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        d2 = nn.functional.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)
        out = self.final_conv(d2)
        
        return out

# ============================================
# DATASET MODIFICADO PARA GRAYSCALE
# ============================================
class FundusSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224, input_mode='grayscale'):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*.*g')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*.*g')))
        self.input_mode = input_mode
        
        print(f"📂 Imagens encontradas: {len(self.images_paths)}")
        print(f"📂 Máscaras encontradas: {len(self.masks_paths)}")
        print(f"📂 Modo de entrada: {input_mode}")
        
        img_names = {os.path.basename(p).lower(): p for p in self.images_paths}
        mask_names = {os.path.basename(p).lower(): p for p in self.masks_paths}
        
        self.valid_pairs = []
        for name, img_path in img_names.items():
            if name in mask_names:
                self.valid_pairs.append((img_path, mask_names[name]))
            else:
                base_name = os.path.splitext(name)[0]
                for mask_name, mask_path in mask_names.items():
                    if os.path.splitext(mask_name)[0] == base_name:
                        self.valid_pairs.append((img_path, mask_path))
                        break
        
        print(f"✅ Pares válidos encontrados: {len(self.valid_pairs)}")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("Nenhum par de imagem-máscara encontrado!")
        
        self.img_size = img_size
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        # Carregar diretamente em tons de cinza (sem conversão)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {img_path}")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Erro ao carregar máscara: {mask_path}")
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

# ============================================
# LOSS FUNCTION
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

# ============================================
# MÉTRICAS
# ============================================
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
# FUNÇÕES DE TREINAMENTO COM TEMPO
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
        
        batch_metrics = compute_metrics(outputs.detach(), masks.detach())
        for k in metrics:
            metrics[k] += batch_metrics[k]
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
        progress_bar = tqdm(val_loader, desc='Validation')
        for images, masks in progress_bar:
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
# FUNÇÃO DE TREINAMENTO (COM SALVAMENTO DO MELHOR MODELO)
# ============================================
def train_model(model, train_loader, val_loader, config, run_id=0):
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', 
        patience=config.scheduler_patience, 
        factor=config.scheduler_factor
    )
    
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    stopped_epoch = config.epochs
    
    # Guardar o melhor estado do modelo
    best_model_state = None
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_dice': [], 'val_dice': [],
        'train_metrics': [], 'val_metrics': [],
        'train_time': [], 'val_time': []
    }
    
    training_start_time = time.time()
    
    print(f"\n🚀 Treinamento Run {run_id+1}/{config.n_runs} - {config.model_name}")
    print(f"Dispositivo: {config.device}")
    print(f"Tamanho da imagem: {config.img_size}x{config.img_size}")
    print(f"Modo de entrada: {config.input_mode} (1 canal)")
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Paciência: {config.patience} épocas")
    print(f"Épocas máximas: {config.epochs}")
    print(f"⏱️ Medição de tempo: {'Ativada' if config.measure_time else 'Desativada'}")
    print(f"💾 Modelo será salvo no final do treinamento (melhor Dice)")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Run {run_id+1} - Época {epoch+1}/{config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
        
        # Atualizar melhor modelo
        if val_metrics['dice'] > best_dice + config.min_delta:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            # Salvar o estado do melhor modelo
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✅ Novo melhor Dice: {best_dice:.4f} (época {epoch+1})")
        else:
            patience_counter += 1
            print(f"⏳ Paciência: {patience_counter}/{config.patience} (melhor Dice: {best_dice:.4f} na época {best_epoch+1})")
        
        if patience_counter >= config.patience:
            stopped_epoch = epoch + 1
            print(f"\n🛑 Early stopping ativado! Parando treinamento na época {stopped_epoch}")
            print(f"Melhor Dice: {best_dice:.4f} (época {best_epoch+1})")
            break
    
    total_training_time = time.time() - training_start_time
    
    # Salvar o melhor modelo no final do treinamento
    if best_model_state is not None:
        model_path = os.path.join(models_dir, f'best_model_run_{run_id}.pth')
        torch.save(best_model_state, model_path)
        print(f"\n✅ Modelo da run {run_id} salvo em: {model_path}")
        print(f"   Melhor Dice: {best_dice:.4f} (época {best_epoch+1})")
    else:
        # Se nenhum modelo melhor foi encontrado, salvar o modelo atual
        model_path = os.path.join(models_dir, f'final_model_run_{run_id}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\n⚠️ Nenhum modelo melhor encontrado, salvando modelo final em: {model_path}")
    
    history['early_stop'] = {
        'stopped_epoch': stopped_epoch,
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'patience_used': patience_counter >= config.patience,
        'total_training_time': total_training_time,
        'model_saved': True,
        'model_path': model_path
    }
    
    if config.save_results:
        save_run_results_csv(history, run_id, config)
    
    return history, best_dice

# ============================================
# FUNÇÃO PARA SALVAR RESULTADOS EM CSV
# ============================================
def save_run_results_csv(history, run_id, config):
    """Salva os resultados de uma execução em CSV (métricas e tempo separados)"""
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # ========== CSV 1: MÉTRICAS ==========
    metrics_data = {
        'epoch': epochs,
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_dice': history['train_dice'],
        'val_dice': history['val_dice'],
    }
    
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        metrics_data[f'train_{metric}'] = [m[metric] for m in history['train_metrics']]
        metrics_data[f'val_{metric}'] = [m[metric] for m in history['val_metrics']]
    
    if history['train_time'] and isinstance(history['train_time'][0], dict):
        for key in history['train_time'][0].keys():
            metrics_data[f'train_time_{key}'] = [t.get(key, 0) if isinstance(t, dict) else 0 for t in history['train_time']]
    
    if history['val_time'] and isinstance(history['val_time'][0], dict):
        for key in history['val_time'][0].keys():
            metrics_data[f'val_time_{key}'] = [t.get(key, 0) if isinstance(t, dict) else 0 for t in history['val_time']]
    
    metrics_data['early_stop_epoch'] = [history['early_stop']['stopped_epoch']] * len(epochs)
    metrics_data['best_epoch'] = [history['early_stop']['best_epoch'] + 1] * len(epochs)
    metrics_data['best_dice'] = [history['early_stop']['best_dice']] * len(epochs)
    metrics_data['model_saved'] = [history['early_stop']['model_saved']] * len(epochs)
    
    df_metrics = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(run_dir, 'metrics_results.csv')
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"✅ Métricas salvas em: {metrics_csv_path}")
    
    # ========== CSV 2: TEMPO ==========
    time_data = {
        'epoch': epochs,
    }
    
    if history['train_time'] and isinstance(history['train_time'][0], dict):
        for key in history['train_time'][0].keys():
            time_data[f'train_{key}'] = [t.get(key, 0) if isinstance(t, dict) else 0 for t in history['train_time']]
    
    if history['val_time'] and isinstance(history['val_time'][0], dict):
        for key in history['val_time'][0].keys():
            time_data[f'val_{key}'] = [t.get(key, 0) if isinstance(t, dict) else 0 for t in history['val_time']]
    
    time_data['total_training_time'] = [history['early_stop']['total_training_time']] * len(epochs)
    
    df_time = pd.DataFrame(time_data)
    time_csv_path = os.path.join(run_dir, 'time_results.csv')
    df_time.to_csv(time_csv_path, index=False)
    print(f"✅ Tempo salvo em: {time_csv_path}")
    
    # ========== CSV 3: RESUMO DA EXECUÇÃO ==========
    summary_data = {
        'run_id': run_id,
        'model_name': config.model_name,
        'img_size': config.img_size,
        'input_mode': config.input_mode,
        'best_val_dice': history['early_stop']['best_dice'],
        'best_epoch': history['early_stop']['best_epoch'] + 1,
        'total_epochs': len(history['train_loss']),
        'early_stopped': history['early_stop']['patience_used'],
        'stopped_epoch': history['early_stop']['stopped_epoch'],
        'total_training_time': history['early_stop']['total_training_time'],
        'final_val_dice': history['val_dice'][-1] if history['val_dice'] else 0,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
        'model_saved': history['early_stop']['model_saved'],
        'model_path': history['early_stop']['model_path']
    }
    
    if history['val_metrics']:
        for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
            summary_data[f'final_val_{metric}'] = history['val_metrics'][-1][metric]
    
    df_summary = pd.DataFrame([summary_data])
    summary_csv_path = os.path.join(run_dir, 'run_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✅ Resumo da execução salvo em: {summary_csv_path}")
    
    return df_metrics, df_time, df_summary

# ============================================
# FUNÇÃO DE TESTE DETALHADA
# ============================================
def test_model_detailed(model, test_loader, device, run_id, config):
    """
    Testa o modelo e retorna métricas detalhadas incluindo:
    - Métricas por imagem (tempo de predição, dice, iou, etc)
    - Métricas agregadas (média, desvio padrão)
    - Tempo total de teste
    """
    model.eval()
    
    per_image_metrics = []
    inference_times = []
    all_metrics = []
    
    total_start_time = time.time()
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc=f'Testing Run {run_id+1}')):
            images = images.to(device)
            
            if config.measure_time:
                start_time = time.time()
            
            outputs = model(images)
            
            if config.measure_time:
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            for i in range(images.size(0)):
                pred = outputs[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                metrics = compute_metrics(pred, mask)
                
                if config.measure_time:
                    per_image_inference_time = inference_time / images.size(0) if config.measure_time else 0
                    metrics['inference_time_per_image'] = float(per_image_inference_time)
                else:
                    metrics['inference_time_per_image'] = 0.0
                
                metrics['image_index'] = idx * test_loader.batch_size + i
                
                per_image_metrics.append(metrics)
                all_metrics.append(metrics)
    
    total_time = time.time() - total_start_time
    
    df_per_image = pd.DataFrame(per_image_metrics)
    
    aggregated_metrics = {}
    for k in df_per_image.columns:
        if k != 'image_index' and pd.api.types.is_numeric_dtype(df_per_image[k]):
            aggregated_metrics[f'{k}_mean'] = float(df_per_image[k].mean())
            aggregated_metrics[f'{k}_std'] = float(df_per_image[k].std())
            aggregated_metrics[f'{k}_min'] = float(df_per_image[k].min())
            aggregated_metrics[f'{k}_max'] = float(df_per_image[k].max())
            aggregated_metrics[f'{k}_median'] = float(df_per_image[k].median())
    
    if config.measure_time and inference_times:
        aggregated_metrics['test_total_time'] = float(total_time)
        aggregated_metrics['test_inference_time_mean'] = float(np.mean(inference_times))
        aggregated_metrics['test_inference_time_std'] = float(np.std(inference_times))
        aggregated_metrics['test_inference_time_min'] = float(np.min(inference_times))
        aggregated_metrics['test_inference_time_max'] = float(np.max(inference_times))
        aggregated_metrics['test_inferences_per_second'] = float(len(inference_times) / total_time)
        aggregated_metrics['test_images_per_second'] = float(len(test_loader.dataset) / total_time)
        aggregated_metrics['test_inference_time_per_image_mean'] = float(df_per_image['inference_time_per_image'].mean())
        aggregated_metrics['test_inference_time_per_image_std'] = float(df_per_image['inference_time_per_image'].std())
    
    if config.save_results:
        per_image_path = os.path.join(test_results_dir, f'per_image_metrics_run_{run_id}.csv')
        df_per_image.to_csv(per_image_path, index=False)
        print(f"✅ Métricas por imagem salvas em: {per_image_path}")
        
        aggregated_path = os.path.join(test_results_dir, f'aggregated_metrics_run_{run_id}.csv')
        df_aggregated = pd.DataFrame([aggregated_metrics])
        df_aggregated.to_csv(aggregated_path, index=False)
        print(f"✅ Métricas agregadas salvas em: {aggregated_path}")
        
        stats_path = os.path.join(test_results_dir, f'test_statistics_run_{run_id}.json')
        with open(stats_path, 'w') as f:
            json.dump(convert_to_serializable(aggregated_metrics), f, indent=4)
        print(f"✅ Estatísticas do teste salvas em: {stats_path}")
    
    return df_per_image, aggregated_metrics

def test_model_average_detailed(config):
    """
    Testa todos os modelos treinados e calcula a média com métricas detalhadas
    """
    test_dataset = FundusSegmentationDataset(
        config.test_images_dir,
        config.test_masks_dir,
        img_size=config.img_size,
        input_mode=config.input_mode
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    all_per_image_dfs = []
    all_aggregated_metrics = []
    
    for run_id in range(config.n_runs):
        print(f"\n📈 Testando Run {run_id+1}/{config.n_runs}")
        
        input_channels = 1
        model = MobileNetV2UNet(num_classes=1, pretrained=False, input_channels=input_channels)
        model_path = os.path.join(models_dir, f'best_model_run_{run_id}.pth')
        
        # Se não encontrar o melhor modelo, tentar o modelo final
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f'final_model_run_{run_id}.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model = model.to(config.device)
            
            df_per_image, aggregated_metrics = test_model_detailed(
                model, test_loader, config.device, run_id, config
            )
            
            df_per_image.insert(0, 'run_id', run_id)
            all_per_image_dfs.append(df_per_image)
            all_aggregated_metrics.append(aggregated_metrics)
        else:
            print(f"⚠️ Modelo da run {run_id} não encontrado em: {model_path}")
    
    if all_per_image_dfs:
        consolidated_per_image = pd.concat(all_per_image_dfs, ignore_index=True)
        consolidated_per_image_path = os.path.join(test_results_dir, 'consolidated_per_image_metrics.csv')
        consolidated_per_image.to_csv(consolidated_per_image_path, index=False)
        print(f"\n✅ Métricas consolidadas por imagem salvas em: {consolidated_per_image_path}")
        
        df_aggregated = pd.DataFrame(all_aggregated_metrics)
        df_aggregated.insert(0, 'run_id', range(len(all_aggregated_metrics)))
        aggregated_summary_path = os.path.join(test_results_dir, 'aggregated_metrics_summary.csv')
        df_aggregated.to_csv(aggregated_summary_path, index=False)
        print(f"✅ Resumo das métricas agregadas salvo em: {aggregated_summary_path}")
        
        aggregated_stats = {}
        for col in df_aggregated.columns:
            if col != 'run_id' and pd.api.types.is_numeric_dtype(df_aggregated[col]):
                aggregated_stats[col] = {
                    'mean': float(df_aggregated[col].mean()),
                    'std': float(df_aggregated[col].std()),
                    'min': float(df_aggregated[col].min()),
                    'max': float(df_aggregated[col].max())
                }
        
        stats_df = pd.DataFrame(aggregated_stats).T
        stats_summary_path = os.path.join(test_results_dir, 'aggregated_statistics_summary.csv')
        stats_df.to_csv(stats_summary_path)
        print(f"✅ Estatísticas consolidadas salvas em: {stats_summary_path}")
        
        print("\n" + "="*50)
        print(f"RESULTADOS DO TESTE - {config.model_name} (Média entre runs)")
        print("="*50)
        
        main_metrics = ['dice_mean', 'iou_mean', 'accuracy_mean', 'sensitivity_mean', 'specificity_mean']
        print("\n📊 MÉTRICAS PRINCIPAIS (Média ± Desvio Padrão):")
        for metric in main_metrics:
            if metric in stats_df.index:
                print(f"  {metric.replace('_mean', '')}: {stats_df.loc[metric, 'mean']:.4f} ± {stats_df.loc[metric, 'std']:.4f}")
        
        time_metrics = ['test_total_time', 'test_inference_time_mean', 'test_inference_time_per_image_mean']
        print("\n⏱️ MÉTRICAS DE TEMPO (Média ± Desvio Padrão):")
        for metric in time_metrics:
            if metric in stats_df.index:
                if 'time' in metric:
                    print(f"  {metric.replace('_', ' ').title()}: {stats_df.loc[metric, 'mean']:.4f}s ± {stats_df.loc[metric, 'std']:.4f}s")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {stats_df.loc[metric, 'mean']:.4f} ± {stats_df.loc[metric, 'std']:.4f}")
        
        if 'test_images_per_second' in stats_df.index:
            print(f"\n📈 Throughput: {stats_df.loc['test_images_per_second', 'mean']:.2f} imagens/s")
        
        print("\n📁 Arquivos gerados em: " + test_results_dir)
        
        return consolidated_per_image, df_aggregated, stats_df
    
    return None, None, None

# ============================================
# FUNÇÃO PARA CRIAR RELATÓRIO CONSOLIDADO FINAL
# ============================================
def create_consolidated_report(config, all_run_summaries, all_metrics_dfs, all_time_dfs):
    """Cria um relatório consolidado final com todas as execuções"""
    
    consolidated_metrics = pd.DataFrame()
    for run_id, df in enumerate(all_metrics_dfs):
        df_copy = df.copy()
        df_copy.insert(0, 'run_id', run_id)
        consolidated_metrics = pd.concat([consolidated_metrics, df_copy], ignore_index=True)
    
    metrics_consolidated_path = os.path.join(reports_dir, f'CONSOLIDADO_METRICAS_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    consolidated_metrics.to_csv(metrics_consolidated_path, index=False)
    print(f"✅ Relatório consolidado de métricas salvo em: {metrics_consolidated_path}")
    
    consolidated_time = pd.DataFrame()
    for run_id, df in enumerate(all_time_dfs):
        df_copy = df.copy()
        df_copy.insert(0, 'run_id', run_id)
        consolidated_time = pd.concat([consolidated_time, df_copy], ignore_index=True)
    
    time_consolidated_path = os.path.join(reports_dir, f'CONSOLIDADO_TEMPO_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    consolidated_time.to_csv(time_consolidated_path, index=False)
    print(f"✅ Relatório consolidado de tempo salvo em: {time_consolidated_path}")
    
    summary_df = pd.DataFrame(all_run_summaries)
    summary_consolidated_path = os.path.join(reports_dir, f'RESUMO_EXECUCOES_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    summary_df.to_csv(summary_consolidated_path, index=False)
    print(f"✅ Resumo consolidado das execuções salvo em: {summary_consolidated_path}")
    
    final_metrics_cols = ['final_val_dice', 'final_val_loss', 'final_val_accuracy', 
                          'final_val_iou', 'final_val_sensitivity', 'final_val_specificity']
    
    stats_data = {}
    for col in final_metrics_cols:
        if col in summary_df.columns:
            stats_data[col] = {
                'mean': summary_df[col].mean(),
                'std': summary_df[col].std(),
                'min': summary_df[col].min(),
                'max': summary_df[col].max(),
                'median': summary_df[col].median()
            }
    
    time_cols = ['total_training_time']
    for col in time_cols:
        if col in summary_df.columns:
            stats_data[col] = {
                'mean': summary_df[col].mean(),
                'std': summary_df[col].std(),
                'min': summary_df[col].min(),
                'max': summary_df[col].max(),
                'median': summary_df[col].median()
            }
    
    stats_data['best_val_dice'] = {
        'mean': summary_df['best_val_dice'].mean(),
        'std': summary_df['best_val_dice'].std(),
        'min': summary_df['best_val_dice'].min(),
        'max': summary_df['best_val_dice'].max(),
        'median': summary_df['best_val_dice'].median()
    }
    
    stats_df = pd.DataFrame(stats_data).T
    stats_path = os.path.join(reports_dir, f'ESTATISTICAS_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    stats_df.to_csv(stats_path)
    print(f"✅ Estatísticas descritivas salvas em: {stats_path}")
    
    print("\n" + "="*70)
    print("📊 RESUMO FINAL DAS EXECUÇÕES")
    print("="*70)
    print(f"\nModelo: {config.model_name}")
    print(f"Número de execuções: {config.n_runs}")
    print(f"Tamanho da imagem: {config.img_size}x{config.img_size}")
    print(f"Modo de entrada: {config.input_mode} (1 canal)")
    print(f"Dispositivo: {config.device}")
    print("\n📈 MÉTRICAS (Média ± Desvio Padrão):")
    for col in ['best_val_dice', 'final_val_dice', 'final_val_iou', 'final_val_accuracy']:
        if col in stats_df.index:
            print(f"  {col}: {stats_df.loc[col, 'mean']:.4f} ± {stats_df.loc[col, 'std']:.4f}")
    
    print("\n⏱️ TEMPO (Média ± Desvio Padrão):")
    if 'total_training_time' in stats_df.index:
        print(f"  Tempo total de treinamento: {stats_df.loc['total_training_time', 'mean']:.2f}s ± {stats_df.loc['total_training_time', 'std']:.2f}s")
    
    print("\n📁 Arquivos gerados:")
    print(f"  - {metrics_consolidated_path}")
    print(f"  - {time_consolidated_path}")
    print(f"  - {summary_consolidated_path}")
    print(f"  - {stats_path}")
    
    return consolidated_metrics, consolidated_time, summary_df, stats_df

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def main():
    print("="*70)
    print(f"{' ' * 20}🚀 {config.model_name}")
    print(f"{' ' * 15}Segmentação de Vasos em Fundoscopia (Grayscale)")
    print("="*70)
    
    print(f"\n📊 CONFIGURAÇÕES:")
    print(f"  Device: {config.device}")
    print(f"  Imagem: {config.img_size}x{config.img_size} (1 canal)")
    print(f"  Modo de entrada: {config.input_mode}")
    print(f"  Execuções: {config.n_runs}")
    print(f"  Épocas máximas: {config.epochs}")
    print(f"  Paciência: {config.patience} épocas")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Medição de tempo: {'Ativada' if config.measure_time else 'Desativada'}")
    print(f"  💾 Modelos serão salvos no final do treinamento")
    print(f"  Diretório: {experiment_dir}")
    
    print("\n📂 Carregando dados...")
    try:
        full_dataset = FundusSegmentationDataset(
            config.train_images_dir, 
            config.train_masks_dir,
            img_size=config.img_size,
            input_mode=config.input_mode
        )
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return
    
    if len(full_dataset) < 10:
        print(f"❌ Poucos dados: {len(full_dataset)} imagens. Mínimo necessário: 10")
        return
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if train_size < 2 or val_size < 2:
        print(f"❌ Dados insuficientes para treino/validação: Treino={train_size}, Val={val_size}")
        return
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"  Treino: {len(train_dataset)} imagens (grayscale)")
    print(f"  Validação: {len(val_dataset)} imagens (grayscale)")
    
    all_histories = []
    all_best_dices = []
    all_training_times = []
    all_run_summaries = []
    all_metrics_dfs = []
    all_time_dfs = []
    
    for run_id in range(config.n_runs):
        run_start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"# EXECUÇÃO {run_id+1}/{config.n_runs} - {config.model_name}")
        print(f"{'#'*70}")
        
        try:
            input_channels = 1
            model = MobileNetV2UNet(
                num_classes=config.num_classes, 
                pretrained=config.pretrained,
                input_channels=input_channels
            )
            model = model.to(config.device)
            
            test_input = torch.randn(1, input_channels, 224, 224).to(config.device)
            test_output = model(test_input)
            print(f"✅ Teste forward pass - Input: {test_input.shape}, Output: {test_output.shape}")
        except Exception as e:
            print(f"❌ Erro ao criar modelo: {e}")
            continue
        
        history, best_dice = train_model(model, train_loader, val_loader, config, run_id)
        all_histories.append(history)
        all_best_dices.append(best_dice)
        
        run_total_time = time.time() - run_start_time
        all_training_times.append(run_total_time)
        
        if config.save_results:
            df_metrics, df_time, df_summary = save_run_results_csv(history, run_id, config)
            all_metrics_dfs.append(df_metrics)
            all_time_dfs.append(df_time)
            all_run_summaries.append(df_summary.iloc[0].to_dict())
        
        print(f"\n⏱️ Tempo total da execução {run_id+1}: {run_total_time:.2f}s")
        
        del model
        torch.cuda.empty_cache()
    
    if all_run_summaries and all_metrics_dfs and all_time_dfs:
        print("\n" + "="*70)
        print("📊 CRIANDO RELATÓRIO CONSOLIDADO FINAL")
        print("="*70)
        create_consolidated_report(config, all_run_summaries, all_metrics_dfs, all_time_dfs)
    
    if all_best_dices:
        print("\n" + "="*70)
        print("📊 CALCULANDO MÉDIAS ENTRE EXECUÇÕES")
        print("="*70)
        
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
    
    print("\n" + "="*70)
    print("🧪 INICIANDO TESTE DOS MODELOS")
    print("="*70)
    
    if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
        test_df, test_agg_df, test_stats_df = test_model_average_detailed(config)
    
    print("\n" + "="*70)
    print(f"✅ EXPERIMENTO CONCLUÍDO - {config.model_name}")
    print("="*70)
    print(f"\n📁 Resultados individuais: {experiment_dir}")
    print(f"📁 Modelos salvos: {models_dir}")
    print(f"📁 Relatórios consolidados: {reports_dir}")
    print(f"📁 Resultados de teste: {test_results_dir}")
    
    if all_training_times:
        print(f"\n⏱️ RESUMO DE TEMPO:")
        print(f"  Tempo total médio por execução: {np.mean(all_training_times):.2f}s")
        print(f"  Tempo total de todas as execuções: {np.sum(all_training_times):.2f}s")

if __name__ == "__main__":
    main()