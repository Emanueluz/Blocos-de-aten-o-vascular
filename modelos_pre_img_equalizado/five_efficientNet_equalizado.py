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
    train_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/equalizado'
    train_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth'
    test_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/test/equalizado'
    test_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth'
    
    num_classes = 1
    img_size = 224
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    
    # NOVAS CONFIGURAÇÕES
    n_runs = 5
    save_results = True
    results_dir = './results-pre-equalizado'
    model_name = 'EfficientNetB0_UNet_equalizado'
    experiment_name = f'{model_name}_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}'
    
    # Early Stopping
    patience = 10
    min_delta = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = './best_EfficientNetB0UNet_equalizado_segmentation.pth'
    
    # Outros parâmetros
    num_workers = 4
    pin_memory = True if torch.cuda.is_available() else False
    pretrained = True
    scheduler_patience = 5
    scheduler_factor = 0.5
    
    # NOVA CONFIGURAÇÃO: Modo de entrada
    input_mode = 'grayscale'  # 'grayscale' ou 'rgb'

config = Config()

# Criar diretórios
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

# ============================================
# FUNÇÃO AUXILIAR
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
    else:
        return obj

# ============================================
# MODELO EfficientNetB0 UNet MODIFICADO PARA GRAYSCALE
# ============================================
class EfficientNetB0UNet(nn.Module):
    """
    U-Net com encoder EfficientNet B0
    Modificado para aceitar imagens em tons de cinza (1 canal)
    """
    def __init__(self, num_classes=1, pretrained=True, input_channels=1):
        super(EfficientNetB0UNet, self).__init__()
        
        self.input_channels = input_channels
        
        # Carregar EfficientNet B0
        if pretrained:
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            efficientnet = models.efficientnet_b0(weights=None)
        
        features = efficientnet.features
        
        # MODIFICAÇÃO: Ajustar a primeira camada convolucional para aceitar 1 canal
        # A EfficientNet espera 3 canais (RGB), mas temos 1 canal (grayscale)
        # Então substituímos a primeira convolução
        first_conv = features[0][0]  # Primeira camada conv2d
        
        # Criar nova convolução com 1 canal de entrada
        # Mantendo o mesmo número de filtros de saída (32) e kernel size
        new_conv = nn.Conv2d(
            in_channels=input_channels,  # 1 canal (grayscale)
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # Inicializar os pesos da nova convolução
        # Se for pretrained, queremos aproveitar os pesos do RGB
        if pretrained:
            # Média dos pesos dos 3 canais RGB para 1 canal
            with torch.no_grad():
                # Pesos originais: (out_channels, 3, k, k)
                original_weights = first_conv.weight
                # Calcular a média ao longo do canal de entrada (dim=1)
                new_weights = original_weights.mean(dim=1, keepdim=True)
                new_conv.weight.data = new_weights
                
                # Copiar o bias se existir
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data
        else:
            # Inicialização padrão se não for pretrained
            new_conv.reset_parameters()
        
        # Substituir a primeira convolução
        features[0][0] = new_conv
        
        # EfficientNet B0 channels reais:
        # Block 1: 32 canais
        # Block 2: 32 canais  
        # Block 3: 40 canais
        # Block 4: 80 canais
        # Block 5: 112 canais
        # Block 6: 192 canais
        # Block 7: 320 canais
        # Block 8: 1280 canais (final)
        
        self.enc1 = features[0:1]      # 32 canais, 112x112
        self.enc2 = features[1:2]      # 16 canais, 56x56
        self.enc3 = features[2:4]      # 40 canais, 28x28
        self.enc4 = features[4:6]      # 112 canais, 14x14
        self.enc5 = features[6:8]      # 320 canais, 7x7
        
        # Center (bottleneck)
        self.center = nn.Sequential(
            nn.Conv2d(320, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - com canais ajustados para EfficientNet
        self.up5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128 + 112, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64 + 40, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, padding=1),
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
        e1 = self.enc1(x)   # 32 x 112 x 112
        e2 = self.enc2(e1)  # 16 x 56 x 56
        e3 = self.enc3(e2)  # 40 x 28 x 28
        e4 = self.enc4(e3)  # 112 x 14 x 14
        e5 = self.enc5(e4)  # 320 x 7 x 7
        
        center = self.center(e5)  # 256 x 7 x 7
        
        d5 = self.up5(center)     # 128 x 14 x 14
        if d5.shape[2:] != e4.shape[2:]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)        # 128 x 14 x 14
        
        d4 = self.up4(d5)         # 64 x 28 x 28
        if d4.shape[2:] != e3.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)        # 64 x 28 x 28
        
        d3 = self.up3(d4)         # 32 x 56 x 56
        if d3.shape[2:] != e2.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)        # 32 x 56 x 56
        
        d2 = self.up2(d3)         # 16 x 112 x 112
        if d2.shape[2:] != e1.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)        # 16 x 112 x 112
        
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
        
        # Verificar correspondência
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
        
        # Carregar imagem em tons de cinza
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {img_path}")
        
        # Carregar máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Erro ao carregar máscara: {mask_path}")
        
        # Redimensionar
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Normalizar
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        # MODIFICAÇÃO: Converter para tensor
        # Imagem: (H, W) -> (1, H, W) para grayscale
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        
        # Se quiser converter para RGB (3 canais) para usar com modelos pré-treinados
        if self.input_mode == 'rgb':
            # Repetir o canal 3 vezes para simular RGB
            image = image.repeat(3, 1, 1)  # (3, H, W)
        
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
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
    """Calcula métricas para segmentação"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Aplicar sigmoid se for logits
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
# FUNÇÕES DE TREINAMENTO
# ============================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    progress_bar = tqdm(train_loader, desc='Training')
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        batch_metrics = compute_metrics(outputs.detach(), masks.detach())
        for k in metrics:
            metrics[k] += batch_metrics[k]
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    num_batches = len(train_loader)
    avg_loss = running_loss / num_batches
    for k in metrics:
        metrics[k] /= num_batches
    
    return avg_loss, metrics

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            batch_metrics = compute_metrics(outputs, masks)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    
    num_batches = len(val_loader)
    avg_loss = running_loss / num_batches
    for k in metrics:
        metrics[k] /= num_batches
    
    return avg_loss, metrics

# ============================================
# FUNÇÃO DE TREINAMENTO COM EARLY STOPPING E RELATÓRIOS
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
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_dice': [], 'val_dice': [],
        'train_metrics': [], 'val_metrics': []
    }
    
    print(f"\n🚀 Treinamento Run {run_id+1}/{config.n_runs} - {config.model_name}")
    print(f"Dispositivo: {config.device}")
    print(f"Tamanho da imagem: {config.img_size}x{config.img_size}")
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Paciência: {config.patience} épocas")
    print(f"Épocas máximas: {config.epochs}\n")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Run {run_id+1} - Época {epoch+1}/{config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config.device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        print(f"\n📊 Treino - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"📊 Validação - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
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
            print(f"\n🛑 Early stopping ativado! Parando treinamento na época {stopped_epoch}")
            print(f"Melhor Dice: {best_dice:.4f} (época {best_epoch+1})")
            break
    
    history['early_stop'] = {
        'stopped_epoch': stopped_epoch,
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'patience_used': patience_counter >= config.patience
    }
    
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
    
    # Criar DataFrame
    data = {
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_dice': history['train_dice'],
        'val_dice': history['val_dice'],
    }
    
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        data[f'train_{metric}'] = [m[metric] for m in history['train_metrics']]
        data[f'val_{metric}'] = [m[metric] for m in history['val_metrics']]
    
    df = pd.DataFrame(data)
    
    csv_path = os.path.join(run_dir, 'training_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Resultados salvos em: {csv_path}")
    
    # Salvar métricas finais
    final_metrics = {
        'run_id': int(run_id),
        'model_name': config.model_name,
        'img_size': config.img_size,
        'input_mode': config.input_mode,
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
    }
    
    for metric in ['accuracy', 'iou', 'sensitivity', 'specificity']:
        final_metrics[f'final_val_{metric}'] = float(history['val_metrics'][-1][metric])
        final_metrics[f'final_train_{metric}'] = float(history['train_metrics'][-1][metric])
    
    final_metrics = convert_to_serializable(final_metrics)
    
    json_path = os.path.join(run_dir, 'final_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    plot_training_history(history, run_id, config)
    
    return df, final_metrics

def plot_training_history(history, run_id, config):
    """Plota e salva os gráficos de treinamento"""
    run_dir = os.path.join(experiment_dir, f'run_{run_id}')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{config.model_name} - Run {run_id+1} (Imagem {config.img_size}x{config.img_size})', 
                 fontsize=16, fontweight='bold')
    
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

# ============================================
# FUNÇÕES PARA RELATÓRIOS CONSOLIDADOS
# ============================================
def compute_average_results(config):
    """Calcula a média dos resultados entre todas as execuções"""
    all_final_metrics = []
    all_best_metrics = []
    
    for run_id in range(config.n_runs):
        run_dir = os.path.join(experiment_dir, f'run_{run_id}')
        json_path = os.path.join(run_dir, 'final_metrics.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                all_final_metrics.append(metrics)
                
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
    
    df_final = pd.DataFrame(all_final_metrics)
    df_best = pd.DataFrame(all_best_metrics)
    
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
    
    stats = convert_to_serializable(stats)
    
    summary_path = os.path.join(experiment_dir, 'summary_results.csv')
    df_final.to_csv(summary_path, index=False)
    
    best_path = os.path.join(experiment_dir, 'best_epochs_results.csv')
    df_best.to_csv(best_path, index=False)
    
    stats_path = os.path.join(experiment_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    create_summary_report(df_final, df_best, stats, config)
    
    return df_final, df_best, stats

def create_summary_report(df_final, df_best, stats, config):
    """Cria um relatório consolidado em texto"""
    report_path = os.path.join(reports_dir, f'RELATORIO_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"{' ' * 30}RELATÓRIO DE EXECUÇÕES\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"📌 MODELO: {config.model_name}\n")
        f.write(f"📅 DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📐 TAMANHO DA IMAGEM: {config.img_size}x{config.img_size}\n")
        f.write(f"🎨 MODO DE ENTRADA: {config.input_mode}\n")
        f.write(f"📊 NÚMERO DE EXECUÇÕES: {config.n_runs}\n")
        f.write(f"🔄 ÉPOCAS MÁXIMAS: {config.epochs}\n")
        f.write(f"⏳ PACIÊNCIA (EARLY STOPPING): {config.patience} épocas\n")
        f.write(f"📈 MIN_DELTA: {config.min_delta}\n")
        f.write(f"💻 DISPOSITIVO: {config.device}\n")
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
            if 'early_stopped' in row:
                f.write(f"    Early Stopping: {'Sim' if row['early_stopped'] else 'Não'}\n")
                f.write(f"    Épocas treinadas: {int(row['total_epochs'])}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write(f"{' ' * 40}FIM DO RELATÓRIO\n")
        f.write("="*100 + "\n")
    
    print(f"✅ Relatório consolidado salvo em: {report_path}")

# ============================================
# FUNÇÃO DE TESTE
# ============================================
def test_model(model, test_loader, device, run_id, config):
    """Testa o modelo e salva os resultados"""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f'Testing Run {run_id+1}'):
            images = images.to(device)
            outputs = model(images)
            
            metrics = compute_metrics(outputs.cpu().numpy(), masks.cpu().numpy())
            all_metrics.append(metrics)
    
    final_metrics = {}
    for k in all_metrics[0].keys():
        final_metrics[k] = float(np.mean([m[k] for m in all_metrics]))
    
    if config.save_results:
        run_dir = os.path.join(experiment_dir, f'run_{run_id}')
        os.makedirs(run_dir, exist_ok=True)
        
        test_df = pd.DataFrame(all_metrics)
        test_df.to_csv(os.path.join(run_dir, 'test_results.csv'), index=False)
        
        test_metrics_path = os.path.join(run_dir, 'test_metrics.json')
        with open(test_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
    
    return final_metrics

def test_model_average(config):
    """Testa todos os modelos treinados e calcula a média"""
    test_dataset = FundusSegmentationDataset(
        config.test_images_dir,
        config.test_masks_dir,
        img_size=config.img_size,
        input_mode=config.input_mode
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    all_test_metrics = []
    
    for run_id in range(config.n_runs):
        print(f"\n📈 Testando Run {run_id+1}/{config.n_runs}")
        
        # Determinar número de canais de entrada
        input_channels = 1 if config.input_mode == 'grayscale' else 3
        model = EfficientNetB0UNet(num_classes=1, pretrained=config.pretrained, input_channels=input_channels)
        model_path = f"{config.best_model_path}_run{run_id}"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            model = model.to(config.device)
            
            metrics = test_model(model, test_loader, config.device, run_id, config)
            all_test_metrics.append(metrics)
        else:
            print(f"⚠️ Modelo da run {run_id} não encontrado!")
    
    if all_test_metrics:
        test_df = pd.DataFrame(all_test_metrics)
        test_summary_path = os.path.join(experiment_dir, 'test_summary.csv')
        test_df.to_csv(test_summary_path, index=False)
        
        test_stats = {}
        for col in test_df.columns:
            test_stats[col] = {
                'mean': float(test_df[col].mean()),
                'std': float(test_df[col].std()),
                'min': float(test_df[col].min()),
                'max': float(test_df[col].max())
            }
        
        test_stats = convert_to_serializable(test_stats)
        
        test_stats_path = os.path.join(experiment_dir, 'test_statistics.json')
        with open(test_stats_path, 'w') as f:
            json.dump(test_stats, f, indent=4)
        
        print("\n" + "="*50)
        print(f"RESULTADOS DO TESTE - {config.model_name} (Média entre runs)")
        print("="*50)
        for metric, values in test_stats.items():
            print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        return test_df, test_stats
    
    return None, None

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def main():
    print("="*70)
    print(f"{' ' * 20}🚀 {config.model_name}")
    print(f"{' ' * 15}Segmentação de Vasos em Fundoscopia")
    print("="*70)
    
    print(f"\n📊 CONFIGURAÇÕES:")
    print(f"  Device: {config.device}")
    print(f"  Imagem: {config.img_size}x{config.img_size}")
    print(f"  Modo de entrada: {config.input_mode}")
    print(f"  Execuções: {config.n_runs}")
    print(f"  Épocas máximas: {config.epochs}")
    print(f"  Paciência: {config.patience} épocas")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Diretório: {experiment_dir}")
    
    # Preparar dados
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
    
    print(f"  Treino: {len(train_dataset)} imagens")
    print(f"  Validação: {len(val_dataset)} imagens")
    
    # Executar múltiplos treinamentos
    all_histories = []
    all_best_dices = []
    
    for run_id in range(config.n_runs):
        print(f"\n{'#'*70}")
        print(f"# EXECUÇÃO {run_id+1}/{config.n_runs} - {config.model_name}")
        print(f"{'#'*70}")
        
        try:
            # Determinar número de canais de entrada
            input_channels = 1 if config.input_mode == 'grayscale' else 3
            
            model = EfficientNetB0UNet(
                num_classes=config.num_classes, 
                pretrained=config.pretrained,
                input_channels=input_channels
            )
            model = model.to(config.device)
            
            # Teste rápido
            test_input = torch.randn(1, input_channels, 224, 224).to(config.device)
            test_output = model(test_input)
            print(f"✅ Teste forward pass - Input: {test_input.shape}, Output: {test_output.shape}")
        except Exception as e:
            print(f"❌ Erro ao criar modelo: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        history, best_dice = train_model(model, train_loader, val_loader, config, run_id)
        all_histories.append(history)
        all_best_dices.append(best_dice)
        
        if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
            try:
                test_dataset = FundusSegmentationDataset(
                    config.test_images_dir,
                    config.test_masks_dir,
                    img_size=config.img_size,
                    input_mode=config.input_mode
                )
                test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
                
                model_path = f"{config.best_model_path}_run{run_id}"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=config.device))
                    model = model.to(config.device)
                    test_model(model, test_loader, config.device, run_id, config)
            except Exception as e:
                print(f"⚠️ Erro no teste: {e}")
        
        del model
        torch.cuda.empty_cache()
    
    # Calcular médias
    if all_best_dices:
        print("\n" + "="*70)
        print("📊 CALCULANDO MÉDIAS ENTRE EXECUÇÕES")
        print("="*70)
        
        best_dice_array = np.array(all_best_dices)
        print(f"\nMelhor Dice por execução:")
        for i, dice in enumerate(all_best_dices):
            print(f"  Run {i+1}: {dice:.4f}")
        print(f"\nMédia dos melhores Dices: {best_dice_array.mean():.4f} ± {best_dice_array.std():.4f}")
        
        compute_average_results(config)
        
        if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
            test_df, test_stats = test_model_average(config)
    
    print("\n" + "="*70)
    print(f"✅ EXPERIMENTO CONCLUÍDO - {config.model_name}")
    print("="*70)
    print(f"\n📁 Resultados salvos em: {experiment_dir}")
    print(f"📁 Relatórios consolidados em: {reports_dir}")

if __name__ == "__main__":
    main()