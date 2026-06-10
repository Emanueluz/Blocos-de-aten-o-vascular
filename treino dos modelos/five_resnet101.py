import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Dados
    train_images_dir = './fundus_images/train/images'
    train_masks_dir = './fundus_images/train/masks'
    val_images_dir = './fundus_images/val/images'
    val_masks_dir = './fundus_images/val/masks'
    test_images_dir = './fundus_images/test/images'
    test_masks_dir = './fundus_images/test/masks'
    
    # Parâmetros da imagem
    img_height = 512
    img_width = 512
    in_channels = 3  # RGB
    num_classes = 1  # Segmentação binária: veia (1) ou fundo (0)
    
    # Treinamento
    batch_size = 8  # Ajuste conforme sua GPU
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    checkpoint_dir = './checkpoints_segmentation'
    best_model_path = './best_resnet101_unet_fundus.pth'
    
    # Logging
    log_interval = 10

config = Config()
os.makedirs(config.checkpoint_dir, exist_ok=True)

# ============================================
# ARQUITETURA RESNET101 COMO BACKBONE DA UNET
# ============================================

class ResNet101UNet(nn.Module):
    """
    UNet com encoder ResNet101 pré-treinado para segmentação de vasos
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNet101UNet, self).__init__()
        
        # Carregar ResNet101 pré-treinado
        import torchvision.models as models
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Encoder (downsampling) - Pegando camadas intermediárias
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # Output: 64 canais, resolução reduzida pela metade
        
        self.encoder2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )  # Output: 256 canais
        
        self.encoder3 = resnet.layer2  # Output: 512 canais
        self.encoder4 = resnet.layer3  # Output: 1024 canais
        self.encoder5 = resnet.layer4  # Output: 2048 canais
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling) com conexões skip
        self.upconv5 = self._make_decoder_block(2048 + 1024, 512)
        self.upconv4 = self._make_decoder_block(1024 + 512, 256)
        self.upconv3 = self._make_decoder_block(512 + 256, 128)
        self.upconv2 = self._make_decoder_block(256 + 128, 64)
        self.upconv1 = self._make_decoder_block(128 + 64, 64)
        
        # Camada final de classificação
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()  # Para segmentação binária
        )
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder com salvamento das skip connections
        enc1 = self.encoder1(x)      # (B, 64, H/2, W/2)
        enc2 = self.encoder2(enc1)   # (B, 256, H/4, W/4)
        enc3 = self.encoder3(enc2)   # (B, 512, H/8, W/8)
        enc4 = self.encoder4(enc3)   # (B, 1024, H/16, W/16)
        enc5 = self.encoder5(enc4)   # (B, 2048, H/32, W/32)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)  # (B, 1024, H/32, W/32)
        
        # Decoder com upsampling e skip connections
        # Upsample 1: bottleneck + enc5
        up5 = nn.functional.interpolate(bottleneck, size=enc5.shape[2:], mode='bilinear', align_corners=True)
        up5 = torch.cat([up5, enc5], dim=1)
        dec5 = self.upconv5(up5)  # (B, 512, H/32, W/32)
        
        # Upsample 2: dec5 + enc4
        up4 = nn.functional.interpolate(dec5, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        up4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.upconv4(up4)  # (B, 256, H/16, W/16)
        
        # Upsample 3: dec4 + enc3
        up3 = nn.functional.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.upconv3(up3)  # (B, 128, H/8, W/8)
        
        # Upsample 4: dec3 + enc2
        up2 = nn.functional.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.upconv2(up2)  # (B, 64, H/4, W/4)
        
        # Upsample 5: dec2 + enc1
        up1 = nn.functional.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.upconv1(up1)  # (B, 64, H/2, W/2)
        
        # Upsample final para resolução original
        final = nn.functional.interpolate(dec1, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Classificação final
        output = self.final_conv(final)  # (B, num_classes, H, W)
        
        return output

# ============================================
# DATASET PARA FUNDOSCOPIA COM MÁSCARAS
# ============================================

class FundusSegmentationDataset(Dataset):
    """
    Dataset para imagens de fundoscopia e máscaras de vasos
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Listar todas as imagens
        self.images = sorted([f for f in os.listdir(images_dir) 
                              if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar máscara (deve ter o mesmo nome)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.tif', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalizar máscara para 0 e 1
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        
        # Aplicar transformações
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0).float()  # (1, H, W)

# ============================================
# TRANSFORMAÇÕES COM ALBUMENTATIONS
# ============================================

def get_transforms(img_size=(512, 512)):
    """
    Transformações para treino e validação
    """
    # Transformações para treino (com augmentation)
    train_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Transformações para validação/teste (sem augmentation)
    val_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

# ============================================
# FUNÇÕES DE LOSS E MÉTRICAS
# ============================================

class CombinedLoss(nn.Module):
    """
    Combinação de Binary Cross Entropy e Dice Loss para segmentação
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        
    def dice_loss(self, pred, target):
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calcula métricas de segmentação
    """
    pred_binary = (pred > threshold).float()
    
    # Aplanar tensores
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Métricas
    accuracy = accuracy_score(target_flat, pred_flat)
    iou = jaccard_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    return accuracy, iou, f1

# ============================================
# VISUALIZAÇÃO DE RESULTADOS
# ============================================

def visualize_predictions(model, dataloader, device, num_samples=4):
    """
    Visualiza predições do modelo
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            image = image.to(device)
            mask = mask.to(device)
            
            # Predição
            pred = model(image)
            
            # Desnormalizar imagem para visualização
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_display = (image * std + mean).cpu().squeeze().permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Plotar
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Imagem Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.cpu().squeeze(), cmap='gray')
            axes[i, 1].set_title('Máscara Real')
            axes[i, 1].axis('off')
            
            pred_display = pred.cpu().squeeze().numpy()
            axes[i, 2].imshow(pred_display, cmap='gray')
            axes[i, 2].set_title('Predição')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150)
    plt.show()

# ============================================
# LOOP DE TREINAMENTO
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images, masks = images.to(device), masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métricas
        acc, iou, f1 = calculate_metrics(outputs, masks)
        running_loss += loss.item()
        running_acc += acc
        running_iou += iou
        
        # Atualizar barra
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': acc,
            'iou': iou
        })
    
    return (running_loss / len(train_loader), 
            running_acc / len(train_loader), 
            running_iou / len(train_loader))

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            acc, iou, f1 = calculate_metrics(outputs, masks)
            running_loss += loss.item()
            running_acc += acc
            running_iou += iou
    
    return (running_loss / len(val_loader), 
            running_acc / len(val_loader), 
            running_iou / len(val_loader))

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main():
    print("🚀 Iniciando segmentação de vasos em fundoscopia com ResNet101-UNet")
    print(f"Device: {config.device}")
    
    # 1. Preparar dados
    print("\n📂 Carregando datasets...")
    train_transform, val_transform = get_transforms((config.img_height, config.img_width))
    
    train_dataset = FundusSegmentationDataset(
        config.train_images_dir, config.train_masks_dir, transform=train_transform
    )
    val_dataset = FundusSegmentationDataset(
        config.val_images_dir, config.val_masks_dir, transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 2. Construir modelo
    print("\n🏗️ Construindo modelo...")
    model = ResNet101UNet(num_classes=config.num_classes, pretrained=True)
    model = model.to(config.device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 3. Configurar otimização
    criterion = CombinedLoss(bce_weight=0.4, dice_weight=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    
    # 4. Loop de treinamento
    print("\n🎯 Iniciando treinamento...")
    best_val_iou = 0.0
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'train_iou': [],
               'val_loss': [], 'val_acc': [], 'val_iou': []}
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 50)
        
        # Treinar
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        # Validar
        val_loss, val_acc, val_iou = validate_epoch(
            model, val_loader, criterion, config.device
        )
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)
        
        # Print resultados
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | IoU: {val_iou:.4f}")
        
        # Salvar melhor modelo
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, config.best_model_path)
            print(f"✅ Melhor modelo salvo! IoU: {val_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 20:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
        
        # Visualizar predições a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            visualize_predictions(model, val_loader, config.device, num_samples=3)
    
    print(f"\n✅ Treinamento concluído! Melhor IoU: {best_val_iou:.4f}")
    
    # Visualizar resultados finais
    visualize_predictions(model, val_loader, config.device, num_samples=5)
    
    # Plotar histórico
    plot_training_history(history)

def plot_training_history(history):
    """
    Plota curvas de treinamento
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    # IoU (Jaccard Index)
    axes[2].plot(history['train_iou'], label='Train IoU')
    axes[2].plot(history['val_iou'], label='Val IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('IoU Curves')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_segmentation.png', dpi=150)
    plt.show()

# ============================================
# INFERÊNCIA EM UMA NOVA IMAGEM
# ============================================

def segment_vasos(image_path, model_path, device, img_size=(512, 512)):
    """
    Segmenta vasos em uma nova imagem de fundoscopia
    """
    # Carregar modelo
    model = ResNet101UNet(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Carregar e processar imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Transformação
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predição
    with torch.no_grad():
        pred = model(input_tensor)
        pred = pred.squeeze().cpu().numpy()
    
    # Redimensionar para tamanho original
    pred_resized = cv2.resize(pred, (original_size[1], original_size[0]))
    pred_binary = (pred_resized > 0.5).astype(np.uint8) * 255
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(pred_resized, cmap='hot')
    axes[1].set_title('Probabilidade (Hot Map)')
    axes[1].axis('off')
    
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Segmentação Binária')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('vein_segmentation_result.png', dpi=150)
    plt.show()
    
    return pred_binary

# ============================================
# EXECUÇÃO
# ============================================

if __name__ == "__main__":
    # Treinar modelo
    main()
    
    # Exemplo de inferência em nova imagem
    # segment_vasos('nova_fundoscopia.jpg', config.best_model_path, config.device)