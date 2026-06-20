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

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    # Dados
    train_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original'    # imagens originais
    train_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth'      # máscaras de segmentação
    test_images_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Original'
    test_masks_dir = '/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth'
    
    img_size = 224  # ResNet espera 224x224
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_path = './best_resnet101_segmentation.pth'
    
config = Config()
 
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet101UNet(nn.Module):
    """
    U-Net com encoder ResNet101 - VERSÃO FINAL FUNCIONAL
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNet101UNet, self).__init__()
        
        # Carregar ResNet101 pré-treinado
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Encoder
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
        
        # Center
        self.center = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder - TODAS AS DIMENSÕES CORRETAS
        # 7 -> 14
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(256 + 1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 14 -> 28
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 28 -> 56
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 56 -> 112
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 112 -> 224
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # CORREÇÃO: decoder1 agora recebe 16 canais e mantém 16
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 16 -> 16 (não 32!)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Final
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)        # 64 x 112 x 112
        e1_pool = self.maxpool(e1)   # 64 x 56 x 56
        
        e2 = self.encoder2(e1_pool)  # 256 x 56 x 56
        e3 = self.encoder3(e2)       # 512 x 28 x 28
        e4 = self.encoder4(e3)       # 1024 x 14 x 14
        e5 = self.encoder5(e4)       # 2048 x 7 x 7
        
        # Center
        c = self.center(e5)          # 512 x 7 x 7
        
        # Decoder
        d5 = self.up5(c)             # 256 x 14 x 14
        d5 = torch.cat([d5, e4], dim=1)  # 1280 x 14 x 14
        d5 = self.dec5(d5)           # 256 x 14 x 14
        
        d4 = self.up4(d5)            # 128 x 28 x 28
        d4 = torch.cat([d4, e3], dim=1)  # 640 x 28 x 28
        d4 = self.dec4(d4)           # 128 x 28 x 28
        
        d3 = self.up3(d4)            # 64 x 56 x 56
        d3 = torch.cat([d3, e2], dim=1)  # 320 x 56 x 56
        d3 = self.dec3(d3)           # 64 x 56 x 56
        
        d2 = self.up2(d3)            # 32 x 112 x 112
        d2 = torch.cat([d2, e1], dim=1)  # 96 x 112 x 112
        d2 = self.dec2(d2)           # 32 x 112 x 112
        
        d1 = self.up1(d2)            # 16 x 224 x 224
        d1 = self.dec1(d1)           # 16 x 224 x 224 (AGORA FUNCIONA!)
        
        # Saída
        out = self.final(d1)
        out = self.sigmoid(out)
        
        return out
#===========================
# DATASET PERSONALIZADO
# ============================================

class FundusSegmentationDataset(Dataset):
    """
    Dataset para segmentação de vasos em fundoscopia
    """
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
        # Carregar imagem
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar máscara (veias em branco, fundo preto)
        mask = cv2.imread(self.masks_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # Normalizar máscara para 0 e 1
        mask = (mask > 127).astype(np.float32)
        
        # Aplicar transformações
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
# LOSS FUNCTION (Combinação BCE + Dice)
# ============================================

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()
    
    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = self.bce(preds, targets)
        
        # Dice Loss
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        return bce_loss + dice_loss

# ============================================
# MÉTRICAS DE SEGMENTAÇÃO
# ============================================

def compute_metrics(preds, targets, threshold=0.5):
    """
    Calcula métricas para segmentação
    """
    preds_binary = (preds > threshold).astype(np.uint8)
    targets_binary = targets.astype(np.uint8)
    
    # Acurácia
    acc = accuracy_score(targets_binary.flatten(), preds_binary.flatten())
    
    # Dice (F1)
    intersection = (preds_binary & targets_binary).sum()
    dice = (2.0 * intersection) / (preds_binary.sum() + targets_binary.sum() + 1e-6)
    
    # IoU (Jaccard)
    union = (preds_binary | targets_binary).sum()
    iou = intersection / (union + 1e-6)
    
    # Sensibilidade (Recall)
    sens = intersection / (targets_binary.sum() + 1e-6)
    
    # Especificidade
    tn = ((1 - preds_binary) & (1 - targets_binary)).sum()
    spec = tn / ((1 - targets_binary).sum() + 1e-6)
    
    return {
        'accuracy': acc,
        'dice': dice,
        'iou': iou,
        'sensitivity': sens,
        'specificity': spec
    }

# ============================================
# TREINAMENTO
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
        batch_metrics = compute_metrics(
            outputs.detach().cpu().numpy(),
            masks.detach().cpu().numpy()
        )
        for k in metrics:
            metrics[k] += batch_metrics[k]
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Médias
    num_batches = len(train_loader)
    for k in metrics:
        metrics[k] /= num_batches
    
    return running_loss / num_batches, metrics

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            batch_metrics = compute_metrics(
                outputs.cpu().numpy(),
                masks.cpu().numpy()
            )
            for k in metrics:
                metrics[k] += batch_metrics[k]
    
    num_batches = len(val_loader)
    for k in metrics:
        metrics[k] /= num_batches
    
    return running_loss / num_batches, metrics

# ============================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ============================================

def train_model(model, train_loader, val_loader, config):
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    print(f"\n🚀 Treinando no dispositivo: {config.device}")
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        
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
        
        # Print resultados
        print(f"\n📊 Treino - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"📊 Validação - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        # Salvar melhor modelo
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, config.best_model_path)
            print(f"✅ Melhor modelo salvo! Dice: {best_dice:.4f}")
    
    return history

# ============================================
# TESTE E VISUALIZAÇÃO
# ============================================

def test_model(model, test_loader, device):
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            outputs = model(images)
            
            # Métricas
            metrics = compute_metrics(outputs.cpu().numpy(), masks.numpy())
            all_metrics.append(metrics)
            
            # Visualizar primeiras 4 imagens
            if i < 4:
                visualize_segmentation(images.cpu(), masks, outputs.cpu(), i)
    
    # Médias finais
    final_metrics = {}
    for k in all_metrics[0].keys():
        final_metrics[k] = np.mean([m[k] for m in all_metrics])
    
    print("\n" + "="*50)
    print("RESULTADOS DO TESTE")
    print("="*50)
    print(f"Acurácia:  {final_metrics['accuracy']:.4f}")
    print(f"Dice (F1): {final_metrics['dice']:.4f}")
    print(f"IoU:       {final_metrics['iou']:.4f}")
    print(f"Sensibilidade: {final_metrics['sensitivity']:.4f}")
    print(f"Especificidade: {final_metrics['specificity']:.4f}")
    
    return final_metrics

def visualize_segmentation(images, masks, preds, idx):
    """
    Visualiza resultados da segmentação
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(min(4, len(images))):
        # Imagem original
        img = images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original {idx*4+i+1}')
        axes[0, i].axis('off')
        
        # Ground truth
        mask = masks[i].squeeze().numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title('Máscara Real')
        axes[1, i].axis('off')
        
        # Predição
        pred = preds[i].squeeze().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)
        axes[2, i].imshow(pred_binary, cmap='gray')
        axes[2, i].set_title(f'Predição (Dice: {compute_metrics(pred, mask)["dice"]:.3f})')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'segmentation_results_{idx}.png')
    plt.show()

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
    plt.savefig('training_history.png', dpi=300)
    plt.show()

# ============================================
# FUNÇÃO PARA SEGMENTAR UMA NOVA IMAGEM
# ============================================

def segment_fundus_image(model, image_path, device, img_size=224, save_path=None):
    """
    Segmenta vasos em uma nova imagem de fundoscopia
    """
    # Carregar e pré-processar
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    
    # Redimensionar
    image_resized = cv2.resize(image_rgb, (img_size, img_size))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predição
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = prediction.squeeze().cpu().numpy()
    
    # Redimensionar para tamanho original
    prediction_resized = cv2.resize(prediction, (original_size[1], original_size[0]))
    prediction_binary = (prediction_resized > 0.5).astype(np.uint8) * 255
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(prediction_resized, cmap='gray')
    axes[1].set_title('Segmentação (probabilidade)')
    axes[1].axis('off')
    
    axes[2].imshow(prediction_binary, cmap='gray')
    axes[2].set_title('Vasos Segmentados')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Resultado salvo em: {save_path}")
    
    plt.show()
    
    return prediction_binary

# ============================================
# MAIN
# ============================================

def main():
    print("🚀 Treinamento ResNet101 para Segmentação de Vasos em Fundoscopia")
    print(f"Device: {config.device}")
    
    # 1. Carregar datasets
    print("\n📂 Carregando dados...")
    
    # Dividir treino em treino/validação (80/20)
    all_images = sorted(glob(os.path.join(config.train_images_dir, '*.*g')))
    all_masks = sorted(glob(os.path.join(config.train_masks_dir, '*.*g')))
    
    split_idx = int(0.8 * len(all_images))
    
    train_images_dir_temp = config.train_images_dir
    train_masks_dir_temp = config.train_masks_dir
    
    # Dataset completo
    full_dataset = FundusSegmentationDataset(
        config.train_images_dir, 
        config.train_masks_dir,
        img_size=config.img_size
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    print(f"Treino: {len(train_dataset)} imagens")
    print(f"Validação: {len(val_dataset)} imagens")
    
    # 2. Criar modelo
    print("\n🏗️ Construindo modelo ResNet101 para segmentação...")
    model = ResNet101UNet(num_classes=1, pretrained=True)
    model = model.to(config.device)
    
    # 3. Treinar
    print("\n🎯 Iniciando treinamento...")
    history = train_model(model, train_loader, val_loader, config)
    
    # 4. Plotar histórico
    print("\n📊 Plotando curvas de aprendizado...")
    plot_training_history(history)
    
    # 5. Testar (se existir dataset de teste)
    if os.path.exists(config.test_images_dir) and os.path.exists(config.test_masks_dir):
        print("\n📈 Avaliando no conjunto de teste...")
        test_dataset = FundusSegmentationDataset(
            config.test_images_dir,
            config.test_masks_dir,
            img_size=config.img_size
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        test_model(model, test_loader, config.device)
    
    # 6. Exemplo de segmentação em nova imagem
    # segment_fundus_image(model, 'nova_fundoscopia.jpg', config.device)
    
    print("\n✅ Treinamento concluído!")

# ============================================
# TESTE RÁPIDO COM MODELO TREINADO
# ============================================

def test_trained_model(model_path, test_image_path):
    """
    Testa um modelo já treinado em uma imagem
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar modelo
    model = ResNet101UNet(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Modelo carregado! Melhor Dice: {checkpoint.get('best_dice', 'N/A')}")
    
    # Segmentar imagem
    segment_fundus_image(model, test_image_path, device)

if __name__ == "__main__":
    main()
    
    # Exemplo: testar modelo salvo
    # test_trained_model('./best_resnet101_segmentation.pth', 'test_fundus.jpg')