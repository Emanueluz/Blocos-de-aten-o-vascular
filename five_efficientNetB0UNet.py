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
    train_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original'
    train_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth'
    test_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Original'
    test_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth'
    
    num_classes = 1
    img_size = 224  # EfficientNet B0 espera 224x224
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Diretórios
    checkpoint_dir = './checkpoints_EfficientNetB0UNet'
    best_model_path = './best_EfficientNetB0UNet_segmentation.pth'
    logs_dir = './logs_EfficientNetB0UNet'
    
    # Outros parâmetros
    num_workers = 4
    pin_memory = True if torch.cuda.is_available() else False
    pretrained = True
    patience = 15
    scheduler_patience = 5
    scheduler_factor = 0.5

# Criar diretórios
config = Config()
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.logs_dir, exist_ok=True)

print(f"✅ Diretórios criados:")
print(f"   Checkpoints: {config.checkpoint_dir}")
print(f"   Logs: {config.logs_dir}")
print(f"   Device: {config.device}")
print()

# ============================================
# MODELO EfficientNetB0 UNet
# ============================================
class EfficientNetB0UNet(nn.Module):
    """
    U-Net com encoder EfficientNet B0
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetB0UNet, self).__init__()
        
        # Carregar EfficientNet B0
        if pretrained:
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            efficientnet = models.efficientnet_b0(weights=None)
        
        features = efficientnet.features
        
        # EfficientNet B0 channels reais:
        # Block 1: 32 canais
        # Block 2: 32 canais  
        # Block 3: 40 canais
        # Block 4: 80 canais
        # Block 5: 112 canais
        # Block 6: 192 canais
        # Block 7: 320 canais
        # Block 8: 1280 canais (final)
        
        # Encoder - divisão dos blocos do EfficientNet
        # features[0]: Conv2d(3, 32) -> 32 canais
        # features[1]: MBConvBlock(32, 16) -> 16 canais
        # features[2]: MBConvBlock(16, 24) -> 24 canais
        # features[3]: MBConvBlock(24, 40) -> 40 canais
        # features[4]: MBConvBlock(40, 80) -> 80 canais
        # features[5]: MBConvBlock(80, 112) -> 112 canais
        # features[6]: MBConvBlock(112, 192) -> 192 canais
        # features[7]: MBConvBlock(192, 320) -> 320 canais
        # features[8]: Conv2d(320, 1280) -> 1280 canais
        
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
        # 7x7 -> 14x14
        self.up5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128 + 112, 128, 3, padding=1),  # 240 canais
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 14x14 -> 28x28
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64 + 40, 64, 3, padding=1),  # 104 canais
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 28x28 -> 56x56
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, padding=1),  # 48 canais
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 56x56 -> 112x112
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(16 + 32, 16, 3, padding=1),  # 48 canais
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Camada final
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder com skip connections
        e1 = self.enc1(x)   # 32 x 112 x 112
        e2 = self.enc2(e1)  # 16 x 56 x 56
        e3 = self.enc3(e2)  # 40 x 28 x 28
        e4 = self.enc4(e3)  # 112 x 14 x 14
        e5 = self.enc5(e4)  # 320 x 7 x 7
        
        # Center
        center = self.center(e5)  # 256 x 7 x 7
        
        # Decoder com skip connections
        # Nível 5: 7x7 -> 14x14
        d5 = self.up5(center)     # 128 x 14 x 14
        if d5.shape[2:] != e4.shape[2:]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)  # 128 + 112 = 240
        d5 = self.dec5(d5)        # 128 x 14 x 14
        
        # Nível 4: 14x14 -> 28x28
        d4 = self.up4(d5)         # 64 x 28 x 28
        if d4.shape[2:] != e3.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)  # 64 + 40 = 104
        d4 = self.dec4(d4)        # 64 x 28 x 28
        
        # Nível 3: 28x28 -> 56x56
        d3 = self.up3(d4)         # 32 x 56 x 56
        if d3.shape[2:] != e2.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)  # 32 + 16 = 48
        d3 = self.dec3(d3)        # 32 x 56 x 56
        
        # Nível 2: 56x56 -> 112x112
        d2 = self.up2(d3)         # 16 x 112 x 112
        if d2.shape[2:] != e1.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)  # 16 + 32 = 48
        d2 = self.dec2(d2)        # 16 x 112 x 112
        
        # Upsample final para 224x224
        d2 = nn.functional.interpolate(d2, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Saída
        out = self.final_conv(d2)  # num_classes x 224 x 224
        
        return out

# ============================================
# DATASET
# ============================================
class FundusSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*.*g')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*.*g')))
        
        print(f"Encontradas {len(self.images_paths)} imagens e {len(self.masks_paths)} máscaras")
        
        assert len(self.images_paths) == len(self.masks_paths), \
            f"Número diferente: {len(self.images_paths)} imagens vs {len(self.masks_paths)} máscaras"
        
        self.img_size = img_size
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        image = cv2.imread(self.images_paths[idx])
        if image is None:
            raise ValueError(f"Não foi possível carregar: {self.images_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar máscara
        mask = cv2.imread(self.masks_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Não foi possível carregar: {self.masks_paths[idx]}")
        
        # Redimensionar
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Normalizar
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        # Converter para tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
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
        
        # Aplicar sigmoid para o Dice
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
    """
    Calcula métricas para segmentação
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Aplicar sigmoid se for logits
    if preds.min() < 0 or preds.max() > 1:
        preds = 1 / (1 + np.exp(-preds))
    
    preds_binary = (preds > threshold).astype(np.uint8)
    targets_binary = targets.astype(np.uint8)
    
    # Acurácia
    acc = accuracy_score(targets_binary.flatten(), preds_binary.flatten())
    
    # Dice (F1)
    intersection = (preds_binary & targets_binary).sum()
    dice = (2.0 * intersection + 1e-6) / (preds_binary.sum() + targets_binary.sum() + 1e-6)
    
    # IoU (Jaccard)
    union = (preds_binary | targets_binary).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Sensibilidade (Recall)
    sens = (intersection + 1e-6) / (targets_binary.sum() + 1e-6)
    
    # Especificidade
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
        
        # Métricas batch
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
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ============================================
def train_model(model, train_loader, val_loader, config):
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', 
        patience=config.scheduler_patience, 
        factor=config.scheduler_factor
    )
    
    best_dice = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    print(f"\n🚀 Treinando no dispositivo: {config.device}")
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Treino
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Validação
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config.device)
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        
        print(f"\n📊 Treino - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"📊 Validação - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        # Salvar melhor modelo com early stopping
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, config.best_model_path)
            print(f"✅ Melhor modelo salvo! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"⚠️ Early stopping ativado na época {epoch+1}")
                break
    
    return history

# ============================================
# VISUALIZAÇÃO
# ============================================
def plot_training_history(history):
    """
    Plota curvas de loss e Dice
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Dice
    ax2.plot(history['train_dice'], label='Train Dice', marker='o')
    ax2.plot(history['val_dice'], label='Val Dice', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Dice Score Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_EfficientNetB0.png', dpi=300)
    plt.show()

# ============================================
# MAIN
# ============================================
def main():
    print("🚀 Treinamento EfficientNetB0 UNet para Segmentação de Vasos em Fundoscopia")
    print(f"Device: {config.device}")
    
    # Verificar diretórios
    if not os.path.exists(config.train_images_dir):
        print(f"❌ Diretório não encontrado: {config.train_images_dir}")
        return None, None
    
    # 1. Carregar datasets
    print("\n📂 Carregando dados...")
    
    full_dataset = FundusSegmentationDataset(
        config.train_images_dir, 
        config.train_masks_dir,
        img_size=config.img_size
    )
    
    # Divisão treino/validação
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
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
    
    print(f"✅ Treino: {len(train_dataset)} imagens")
    print(f"✅ Validação: {len(val_dataset)} imagens")
    
    # 2. Criar modelo
    print("\n🏗️ Construindo modelo EfficientNetB0 UNet para segmentação...")
    model = EfficientNetB0UNet(num_classes=config.num_classes, pretrained=config.pretrained)
    model = model.to(config.device)
    
    # Teste rápido
    test_input = torch.randn(1, 3, 224, 224).to(config.device)
    test_output = model(test_input)
    print(f"✅ Teste forward pass - Input: {test_input.shape}, Output: {test_output.shape}")
    
    # 3. Treinar
    print("\n🎯 Iniciando treinamento...")
    history = train_model(model, train_loader, val_loader, config)
    
    # 4. Plotar histórico
    if history:
        print("\n📊 Plotando curvas de aprendizado...")
        plot_training_history(history)
    
    print("\n✅ Pipeline de treinamento concluída!")
    
    return model, history

if __name__ == "__main__":
    model, history = main()