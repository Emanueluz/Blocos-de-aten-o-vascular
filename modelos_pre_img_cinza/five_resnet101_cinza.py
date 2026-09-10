#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentação de Vasos em Fundoscopia - ResNet101 UNet (Grayscale)
Uso: python script.py --train_images_dir /path/to/train/images --train_masks_dir /path/to/train/masks --test_dirs /path/to/test1 /path/to/test2 --epochs 50 --runs 5
"""

import argparse
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
import time
import sys

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================
class Config:
    def __init__(self, args):
        # Dados de treino
        self.train_images_dir = args.train_images_dir
        self.train_masks_dir = args.train_masks_dir
        
        # Múltiplas bases de teste
        self.test_datasets = []
        for test_dir in args.test_dirs:
            test_dir = test_dir.strip()
            
            images_path = None
            masks_path = None
            
            # Tenta detectar estruturas comuns automaticamente
            possible_structures = [
                ('Original', 'Ground truth'),
                ('images', 'masks'),
                ('img', 'mask'),
                ('image', 'mask'),
                ('cinza', 'Ground truth'),
                ('gray', 'Ground truth'),
                ('images', 'Ground truth'),
                ('Original', 'masks'),
            ]
            
            for img_sub, mask_sub in possible_structures:
                img_check = os.path.join(test_dir, img_sub)
                mask_check = os.path.join(test_dir, mask_sub)
                if os.path.exists(img_check) and os.path.exists(mask_check):
                    images_path = img_check
                    masks_path = mask_check
                    break
            
            # Fallback: assume que imagens e máscaras estão na mesma pasta
            if images_path is None or masks_path is None:
                images_path = test_dir
                masks_path = test_dir
            
            dataset_name = os.path.basename(os.path.normpath(test_dir))
            self.test_datasets.append({
                'name': dataset_name,
                'images_dir': images_path,
                'masks_dir': masks_path,
                'path': test_dir
            })
        
        # Parâmetros do modelo
        self.num_classes = args.num_classes
        self.img_size = args.img_size
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        
        self.n_runs = args.n_runs
        self.save_results = not args.no_save_results
        self.model_name = args.model_name
        self.experiment_name = f'{self.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        self.patience = args.patience
        self.min_delta = args.min_delta
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        self.best_model_path = './best_resnet101_grayscale.pth'
        
        self.num_workers = args.num_workers
        self.pin_memory = True if torch.cuda.is_available() and not args.no_cuda else False
        self.pretrained = not args.no_pretrained
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        
        self.measure_time = not args.no_measure_time
        self.input_mode = args.input_mode
        self.input_channels = 1 if self.input_mode == 'grayscale' else 3
        
        # Diretórios de saída
        self.results_dir = args.results_dir
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.network_dir = os.path.join(self.results_dir, self.model_name)
        os.makedirs(self.network_dir, exist_ok=True)
        
        self.experiment_dir = os.path.join(self.network_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.models_dir = os.path.join(self.experiment_dir, 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.reports_dir = os.path.join(self.network_dir, 'RELATORIO_DAS_EXECUCOES')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Diretório raiz para resultados de teste (cada base terá subpasta)
        self.test_results_dir = os.path.join(self.experiment_dir, 'test_results')
        os.makedirs(self.test_results_dir, exist_ok=True)

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
# MODELO ResNet101UNet para GRAYSCALE
# ============================================
class ResNet101UNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, input_channels=1):
        super(ResNet101UNet, self).__init__()
        
        self.input_channels = input_channels
        
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        
        first_conv = resnet.conv1
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
                new_weights = original_weights.mean(dim=1, keepdim=True)
                new_conv.weight.data = new_weights
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data
        else:
            new_conv.reset_parameters()
        
        resnet.conv1 = new_conv
        
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
        
        # Decoder
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
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Esperado {self.input_channels} canais, recebeu {x.shape[1]}")
        
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Encoder
        e1 = self.encoder1(x)
        e1_pool = self.maxpool(e1)
        
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Center
        c = self.center(e5)
        
        # Decoder
        d5 = self.up5(c)
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
        
        d1 = self.up1(d2)
        if d1.shape[2:] != (224, 224):
            d1 = nn.functional.interpolate(d1, size=(224, 224), mode='bilinear', align_corners=False)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        out = self.sigmoid(out)
        
        return out

# ============================================
# DATASET
# ============================================
class FundusSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224, input_mode='grayscale'):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*.*g')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*.*g')))
        self.input_mode = input_mode
        
        print(f"  Imagens encontradas: {len(self.images_paths)}")
        print(f"  Mascaras encontradas: {len(self.masks_paths)}")
        print(f"  Modo de entrada: {input_mode}")
        
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
        
        print(f"  Pares validos encontrados: {len(self.valid_pairs)}")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("Nenhum par de imagem-mascara encontrado!")
        
        self.img_size = img_size
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {img_path}")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Erro ao carregar mascara: {mask_path}")
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
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
# TREINO / VALIDACAO
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
# TREINAMENTO
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
    
    best_model_state = None
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_metrics': [], 'val_metrics': [],
        'train_time': [], 'val_time': []
    }
    
    training_start_time = time.time()
    
    print(f"\nTreinamento Run {run_id+1}/{config.n_runs} - {config.model_name}")
    print(f"Dispositivo: {config.device}")
    print(f"Tamanho da imagem: {config.img_size}x{config.img_size}")
    print(f"Modo de entrada: {config.input_mode} ({config.input_channels} canal)")
    print(f"Total de parametros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Paciencia: {config.patience} epocas")
    print(f"Epocas maximas: {config.epochs}")
    print(f"Medicao de tempo: {'Ativada' if config.measure_time else 'Desativada'}")
    print(f"Modelo sera salvo no final do treinamento (melhor Dice)")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Run {run_id+1} - Epoca {epoch+1}/{config.epochs}")
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
        
        print(f"\nTreino - Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"Validacao - Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        if config.measure_time:
            print(f"Treino: {train_time.get('epoch_total_time', 0):.2f}s | "
                  f"Batch: {train_time.get('batch_forward_time_mean', 0):.3f}s")
            print(f"Validacao: {val_time.get('epoch_total_time', 0):.2f}s | "
                  f"Inferencia: {val_time.get('inference_time_mean', 0):.3f}s")
        
        if val_metrics['dice'] > best_dice + config.min_delta:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Novo melhor Dice: {best_dice:.4f} (epoca {epoch+1})")
        else:
            patience_counter += 1
            print(f"Paciencia: {patience_counter}/{config.patience} (melhor Dice: {best_dice:.4f} na epoca {best_epoch+1})")
        
        if patience_counter >= config.patience:
            stopped_epoch = epoch + 1
            print(f"\nEarly stopping ativado! Parando treinamento na epoca {stopped_epoch}")
            print(f"Melhor Dice: {best_dice:.4f} (epoca {best_epoch+1})")
            break
    
    total_training_time = time.time() - training_start_time
    
    if best_model_state is not None:
        model_path = os.path.join(config.models_dir, f'best_model_run_{run_id}.pth')
        torch.save(best_model_state, model_path)
        print(f"\nModelo da run {run_id} salvo em: {model_path}")
        print(f"   Melhor Dice: {best_dice:.4f} (epoca {best_epoch+1})")
    else:
        model_path = os.path.join(config.models_dir, f'final_model_run_{run_id}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\nNenhum modelo melhor encontrado, salvando modelo final em: {model_path}")
    
    history['early_stop'] = {
        'stopped_epoch': stopped_epoch,
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'patience_used': patience_counter >= config.patience,
        'total_training_time': total_training_time,
        'model_saved': True,
        'model_path': model_path
    }
    
    return history, best_dice

# ============================================
# SALVAR RESULTADOS POR RUN
# ============================================
def save_run_results_csv(history, run_id, config):
    run_dir = os.path.join(config.experiment_dir, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
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
    print(f"Metricas salvas em: {metrics_csv_path}")
    
    time_data = {'epoch': epochs}
    
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
    print(f"Tempo salvo em: {time_csv_path}")
    
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
    print(f"Resumo da execucao salvo em: {summary_csv_path}")
    
    return df_metrics, df_time, df_summary

# ============================================
# TESTE DETALHADO (POR BASE)
# ============================================
def test_model_detailed(model, test_loader, device, run_id, config, dataset_name="test"):
    model.eval()
    per_image_metrics = []
    inference_times = []
    
    total_start_time = time.time()
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc=f'Testing {dataset_name} Run {run_id+1}')):
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
                    per_image_inference_time = inference_time / images.size(0)
                    metrics['inference_time_per_image'] = float(per_image_inference_time)
                else:
                    metrics['inference_time_per_image'] = 0.0
                
                metrics['image_index'] = idx * test_loader.batch_size + i
                per_image_metrics.append(metrics)
    
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
        dataset_test_dir = os.path.join(config.test_results_dir, dataset_name)
        os.makedirs(dataset_test_dir, exist_ok=True)
        
        per_image_path = os.path.join(dataset_test_dir, f'per_image_metrics_run_{run_id}.csv')
        df_per_image.to_csv(per_image_path, index=False)
        print(f"Metricas por imagem salvas em: {per_image_path}")
        
        aggregated_path = os.path.join(dataset_test_dir, f'aggregated_metrics_run_{run_id}.csv')
        df_aggregated = pd.DataFrame([aggregated_metrics])
        df_aggregated.to_csv(aggregated_path, index=False)
        print(f"Metricas agregadas salvas em: {aggregated_path}")
        
        stats_path = os.path.join(dataset_test_dir, f'test_statistics_run_{run_id}.json')
        with open(stats_path, 'w') as f:
            json.dump(convert_to_serializable(aggregated_metrics), f, indent=4)
        print(f"Estatisticas do teste salvas em: {stats_path}")
    
    return df_per_image, aggregated_metrics

# ============================================
# TESTE EM TODAS AS BASES (SEPARADAMENTE)
# ============================================
def test_model_average_detailed(config):
    all_results = {}
    
    for dataset_info in config.test_datasets:
        dataset_name = dataset_info['name']
        images_dir = dataset_info['images_dir']
        masks_dir = dataset_info['masks_dir']
        
        print(f"\n{'='*60}")
        print(f"TESTANDO BASE DE DADOS: {dataset_name}")
        print(f"   Imagens: {images_dir}")
        print(f"   Mascaras: {masks_dir}")
        print(f"{'='*60}")
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"Diretorios nao encontrados para {dataset_name}:")
            print(f"   Imagens: {images_dir}")
            print(f"   Mascaras: {masks_dir}")
            continue
        
        try:
            test_dataset = FundusSegmentationDataset(
                images_dir,
                masks_dir,
                img_size=config.img_size,
                input_mode=config.input_mode
            )
        except Exception as e:
            print(f"Erro ao carregar dataset {dataset_name}: {e}")
            continue
        
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        
        all_per_image_dfs = []
        all_aggregated_metrics = []
        
        for run_id in range(config.n_runs):
            print(f"\nTestando Run {run_id+1}/{config.n_runs}")
            
            input_channels = 1 if config.input_mode == 'grayscale' else 3
            model = ResNet101UNet(num_classes=1, pretrained=False, input_channels=input_channels)
            model_path = os.path.join(config.models_dir, f'best_model_run_{run_id}.pth')
            
            if not os.path.exists(model_path):
                model_path = os.path.join(config.models_dir, f'final_model_run_{run_id}.pth')
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=config.device))
                model = model.to(config.device)
                
                df_per_image, aggregated_metrics = test_model_detailed(
                    model, test_loader, config.device, run_id, config, dataset_name
                )
                
                df_per_image.insert(0, 'run_id', run_id)
                all_per_image_dfs.append(df_per_image)
                all_aggregated_metrics.append(aggregated_metrics)
            else:
                print(f"Modelo da run {run_id} nao encontrado em: {model_path}")
        
        if all_per_image_dfs:
            dataset_test_dir = os.path.join(config.test_results_dir, dataset_name)
            os.makedirs(dataset_test_dir, exist_ok=True)
            
            # Consolidado por imagem
            consolidated_per_image = pd.concat(all_per_image_dfs, ignore_index=True)
            consolidated_per_image_path = os.path.join(dataset_test_dir, 'consolidated_per_image_metrics.csv')
            consolidated_per_image.to_csv(consolidated_per_image_path, index=False)
            print(f"\nMetricas consolidadas por imagem salvas em: {consolidated_per_image_path}")
            
            # Resumo agregado (uma linha por run)
            df_aggregated = pd.DataFrame(all_aggregated_metrics)
            df_aggregated.insert(0, 'run_id', range(len(all_aggregated_metrics)))
            aggregated_summary_path = os.path.join(dataset_test_dir, 'aggregated_metrics_summary.csv')
            df_aggregated.to_csv(aggregated_summary_path, index=False)
            print(f"Resumo das metricas agregadas salvo em: {aggregated_summary_path}")
            
            # Estatísticas entre runs
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
            stats_summary_path = os.path.join(dataset_test_dir, 'aggregated_statistics_summary.csv')
            stats_df.to_csv(stats_summary_path)
            print(f"Estatisticas consolidadas salvas em: {stats_summary_path}")
            
            # Relatório final em texto por base
            report_txt_path = os.path.join(dataset_test_dir, 'REPORT.txt')
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"RELATORIO DE TESTE - BASE: {dataset_name}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Modelo: {config.model_name}\n")
                f.write(f"Modo de entrada: {config.input_mode} ({config.input_channels} canal)\n")
                f.write(f"Tamanho da imagem: {config.img_size}x{config.img_size}\n")
                f.write(f"Execucoes testadas: {len(all_aggregated_metrics)}\n\n")
                
                main_metrics = ['dice_mean', 'iou_mean', 'accuracy_mean', 'sensitivity_mean', 'specificity_mean']
                f.write("METRICAS PRINCIPAIS (Media ± Desvio Padrao)\n")
                f.write("-" * 50 + "\n")
                for metric in main_metrics:
                    if metric in stats_df.index:
                        f.write(f"  {metric.replace('_mean', '').capitalize():15s}: "
                                f"{stats_df.loc[metric, 'mean']:.4f} ± {stats_df.loc[metric, 'std']:.4f}\n")
                
                f.write("\nMETRICAS DE TEMPO\n")
                f.write("-" * 50 + "\n")
                time_metrics = ['test_total_time', 'test_inference_time_mean', 'test_inference_time_per_image_mean']
                for metric in time_metrics:
                    if metric in stats_df.index:
                        f.write(f"  {metric.replace('_', ' ').title():30s}: "
                                f"{stats_df.loc[metric, 'mean']:.4f}s ± {stats_df.loc[metric, 'std']:.4f}s\n")
                
                if 'test_images_per_second' in stats_df.index:
                    f.write(f"\n  Throughput: {stats_df.loc['test_images_per_second', 'mean']:.2f} imagens/s\n")
                
                f.write(f"\n{'='*60}\n")
                f.write(f"Fim do relatorio - {dataset_name}\n")
                f.write(f"{'='*60}\n")
            
            print(f"Relatorio em texto salvo em: {report_txt_path}")
            
            print("\n" + "="*50)
            print(f"RESULTADOS DO TESTE - {dataset_name}")
            print("="*50)
            
            print("\nMETRICAS PRINCIPAIS (Media ± Desvio Padrao):")
            for metric in main_metrics:
                if metric in stats_df.index:
                    print(f"  {metric.replace('_mean', '')}: "
                          f"{stats_df.loc[metric, 'mean']:.4f} ± {stats_df.loc[metric, 'std']:.4f}")
            
            print("\nMETRICAS DE TEMPO (Media ± Desvio Padrao):")
            for metric in time_metrics:
                if metric in stats_df.index:
                    if 'time' in metric:
                        print(f"  {metric.replace('_', ' ').title()}: "
                              f"{stats_df.loc[metric, 'mean']:.4f}s ± {stats_df.loc[metric, 'std']:.4f}s")
                    else:
                        print(f"  {metric.replace('_', ' ').title()}: "
                              f"{stats_df.loc[metric, 'mean']:.4f} ± {stats_df.loc[metric, 'std']:.4f}")
            
            if 'test_images_per_second' in stats_df.index:
                print(f"\nThroughput: {stats_df.loc['test_images_per_second', 'mean']:.2f} imagens/s")
            
            all_results[dataset_name] = {
                'per_image': consolidated_per_image,
                'aggregated': df_aggregated,
                'statistics': stats_df
            }
    
    return all_results

# ============================================
# RELATORIO COMPARATIVO ENTRE BASES
# ============================================
def create_cross_dataset_report(config, all_results):
    """
    Cria um relatório CSV/TXT comparando todas as bases de teste lado a lado.
    """
    if not all_results:
        print("Nenhum resultado de teste para comparar.")
        return
    
    comparison_data = []
    
    for dataset_name, result in all_results.items():
        stats_df = result['statistics']
        row = {'dataset': dataset_name}
        
        main_metrics = {
            'dice': 'dice_mean',
            'iou': 'iou_mean',
            'accuracy': 'accuracy_mean',
            'sensitivity': 'sensitivity_mean',
            'specificity': 'specificity_mean',
        }
        
        for name, col in main_metrics.items():
            if col in stats_df.index:
                row[f'{name}_mean'] = stats_df.loc[col, 'mean']
                row[f'{name}_std'] = stats_df.loc[col, 'std']
        
        if 'test_total_time' in stats_df.index:
            row['test_total_time_mean'] = stats_df.loc['test_total_time', 'mean']
        if 'test_inference_time_per_image_mean' in stats_df.index:
            row['inference_time_per_image_mean'] = stats_df.loc['test_inference_time_per_image_mean', 'mean']
        if 'test_images_per_second' in stats_df.index:
            row['throughput_images_per_second'] = stats_df.loc['test_images_per_second', 'mean']
        
        if 'per_image' in result and not result['per_image'].empty:
            n_images = result['per_image'][result['per_image']['run_id'] == 0].shape[0]
            row['num_images'] = n_images
        
        comparison_data.append(row)
    
    df_compare = pd.DataFrame(comparison_data)
    
    cross_dir = os.path.join(config.test_results_dir, 'COMPARATIVO_ENTRE_BASES')
    os.makedirs(cross_dir, exist_ok=True)
    
    csv_path = os.path.join(cross_dir, 'comparativo_entre_bases.csv')
    df_compare.to_csv(csv_path, index=False)
    print(f"\nRelatorio comparativo entre bases salvo em: {csv_path}")
    
    txt_path = os.path.join(cross_dir, 'COMPARATIVO_ENTRE_BASES.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPARATIVO ENTRE BASES DE TESTE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Modelo: {config.model_name}\n")
        f.write(f"Modo de entrada: {config.input_mode}\n")
        f.write(f"Numero de execucoes: {config.n_runs}\n\n")
        
        for _, row in df_compare.iterrows():
            f.write("-"*60 + "\n")
            f.write(f"BASE: {row['dataset']}\n")
            f.write("-"*60 + "\n")
            if 'num_images' in row and pd.notna(row.get('num_images')):
                f.write(f"  Imagens testadas: {int(row['num_images'])}\n")
            for name in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
                if f'{name}_mean' in row and pd.notna(row.get(f'{name}_mean')):
                    f.write(f"  {name.capitalize():15s}: {row[f'{name}_mean']:.4f} ± {row[f'{name}_std']:.4f}\n")
            if 'throughput_images_per_second' in row and pd.notna(row.get('throughput_images_per_second')):
                f.write(f"  Throughput     : {row['throughput_images_per_second']:.2f} imagens/s\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Relatorio comparativo em texto salvo em: {txt_path}")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Comparacao entre Bases de Teste - {config.model_name}', fontsize=14, fontweight='bold')
        
        datasets = df_compare['dataset'].tolist()
        x = np.arange(len(datasets))
        width = 0.25
        
        metrics_to_plot = ['dice_mean', 'iou_mean', 'accuracy_mean']
        labels = ['Dice', 'IoU', 'Accuracy']
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
            if metric in df_compare.columns:
                axes[0].bar(x + i*width, df_compare[metric], width, label=label)
        
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(datasets, rotation=15)
        axes[0].set_ylabel('Valor')
        axes[0].set_title('Metricas Principais por Base')
        axes[0].legend()
        axes[0].grid(True, axis='y')
        
        if 'throughput_images_per_second' in df_compare.columns:
            axes[1].bar(x, df_compare['throughput_images_per_second'], color='steelblue')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(datasets, rotation=15)
            axes[1].set_ylabel('Imagens/segundo')
            axes[1].set_title('Throughput por Base')
            axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(cross_dir, 'comparativo_entre_bases.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Grafico comparativo salvo em: {plot_path}")
    except Exception as e:
        print(f"Nao foi possivel gerar grafico comparativo: {e}")
    
    return df_compare

# ============================================
# RELATORIO CONSOLIDADO (TREINO)
# ============================================
def create_consolidated_report(config, all_run_summaries, all_metrics_dfs, all_time_dfs):
    consolidated_metrics = pd.DataFrame()
    for run_id, df in enumerate(all_metrics_dfs):
        df_copy = df.copy()
        df_copy.insert(0, 'run_id', run_id)
        consolidated_metrics = pd.concat([consolidated_metrics, df_copy], ignore_index=True)
    
    metrics_consolidated_path = os.path.join(config.reports_dir, f'CONSOLIDADO_METRICAS_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    consolidated_metrics.to_csv(metrics_consolidated_path, index=False)
    print(f"Relatorio consolidado de metricas salvo em: {metrics_consolidated_path}")
    
    consolidated_time = pd.DataFrame()
    for run_id, df in enumerate(all_time_dfs):
        df_copy = df.copy()
        df_copy.insert(0, 'run_id', run_id)
        consolidated_time = pd.concat([consolidated_time, df_copy], ignore_index=True)
    
    time_consolidated_path = os.path.join(config.reports_dir, f'CONSOLIDADO_TEMPO_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    consolidated_time.to_csv(time_consolidated_path, index=False)
    print(f"Relatorio consolidado de tempo salvo em: {time_consolidated_path}")
    
    summary_df = pd.DataFrame(all_run_summaries)
    summary_consolidated_path = os.path.join(config.reports_dir, f'RESUMO_EXECUCOES_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    summary_df.to_csv(summary_consolidated_path, index=False)
    print(f"Resumo consolidado das execucoes salvo em: {summary_consolidated_path}")
    
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
    
    for col in ['total_training_time']:
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
    stats_path = os.path.join(config.reports_dir, f'ESTATISTICAS_{config.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    stats_df.to_csv(stats_path)
    print(f"Estatisticas descritivas salvas em: {stats_path}")
    
    print("\n" + "="*70)
    print("RESUMO FINAL DAS EXECUCOES (TREINO)")
    print("="*70)
    print(f"\nModelo: {config.model_name}")
    print(f"Numero de execucoes: {config.n_runs}")
    print(f"Tamanho da imagem: {config.img_size}x{config.img_size}")
    print(f"Modo de entrada: {config.input_mode} ({config.input_channels} canal)")
    print(f"Dispositivo: {config.device}")
    print("\nMETRICAS (Media ± Desvio Padrao):")
    for col in ['best_val_dice', 'final_val_dice', 'final_val_iou', 'final_val_accuracy']:
        if col in stats_df.index:
            print(f"  {col}: {stats_df.loc[col, 'mean']:.4f} ± {stats_df.loc[col, 'std']:.4f}")
    
    print("\nTEMPO (Media ± Desvio Padrao):")
    if 'total_training_time' in stats_df.index:
        print(f"  Tempo total de treinamento: {stats_df.loc['total_training_time', 'mean']:.2f}s ± {stats_df.loc['total_training_time', 'std']:.2f}s")
    
    print("\nArquivos gerados:")
    print(f"  - {metrics_consolidated_path}")
    print(f"  - {time_consolidated_path}")
    print(f"  - {summary_consolidated_path}")
    print(f"  - {stats_path}")
    
    return consolidated_metrics, consolidated_time, summary_df, stats_df

# ============================================
# VISUALIZAÇÃO
# ============================================
def visualize_results(config):
    all_summaries = []
    for run_id in range(config.n_runs):
        run_summary_path = os.path.join(config.experiment_dir, f'run_{run_id}', 'run_summary.csv')
        if os.path.exists(run_summary_path):
            df = pd.read_csv(run_summary_path)
            all_summaries.append(df)
    
    if all_summaries:
        df = pd.concat(all_summaries, ignore_index=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{config.model_name} - Comparacao entre Execucoes', fontsize=16, fontweight='bold')
        
        axes[0, 0].bar(range(len(df)), df['final_val_dice'])
        axes[0, 0].axhline(y=df['final_val_dice'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["final_val_dice"].mean():.4f}')
        axes[0, 0].set_xlabel('Run')
        axes[0, 0].set_ylabel('Dice')
        axes[0, 0].set_title('Dice Final por Execucao')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].bar(range(len(df)), df['final_val_iou'])
        axes[0, 1].axhline(y=df['final_val_iou'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["final_val_iou"].mean():.4f}')
        axes[0, 1].set_xlabel('Run')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('IoU Final por Execucao')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].bar(range(len(df)), df['final_val_accuracy'])
        axes[1, 0].axhline(y=df['final_val_accuracy'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["final_val_accuracy"].mean():.4f}')
        axes[1, 0].set_xlabel('Run')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy Final por Execucao')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].bar(range(len(df)), df['final_val_sensitivity'])
        axes[1, 1].axhline(y=df['final_val_sensitivity'].mean(), color='r', linestyle='--',
                          label=f'Media: {df["final_val_sensitivity"].mean():.4f}')
        axes[1, 1].set_xlabel('Run')
        axes[1, 1].set_ylabel('Sensitivity')
        axes[1, 1].set_title('Sensitivity Final por Execucao')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.reports_dir, f'comparacao_execucoes_{config.model_name}.png'), dpi=300)
        plt.close()
        
        print("\nEstatisticas das execucoes:")
        print(df.describe())

# ============================================
# PARSE ARGUMENTS
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description='ResNet101 UNet para Segmentacao de Vasos em Fundoscopia (Grayscale)')
    
    parser.add_argument('--train_images_dir', required=True, help='Diretorio com imagens de treino')
    parser.add_argument('--train_masks_dir', required=True, help='Diretorio com mascaras de treino')
    parser.add_argument('--test_dirs', nargs='+', required=True,
                        help='Diretorios das bases de teste (ex: /path/Five /path/Reta)')
    
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--input_mode', type=str, default='grayscale', choices=['grayscale', 'rgb'])
    
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--model_name', type=str, default='ResNet101_UNet_Grayscale')
    parser.add_argument('--no_save_results', action='store_true')
    parser.add_argument('--no_measure_time', action='store_true')
    
    return parser.parse_args()

# ============================================
# MAIN
# ============================================
def main():
    args = parse_args()
    config = Config(args)
    
    print("="*70)
    print(f"{' ' * 20} {config.model_name}")
    print(f"{' ' * 15}Segmentacao de Vasos em Fundoscopia ({config.input_mode.upper()})")
    print("="*70)
    
    print("\nDIAGNOSTICO DE GPU")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and not args.no_cuda:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        if args.no_cuda:
            print("CUDA desabilitada por argumento")
        else:
            print("CUDA NAO esta disponivel!")
        print("Usando CPU para treinamento...")
    print("="*50)
    
    print(f"\nCONFIGURACOES:")
    print(f"  Device: {config.device}")
    print(f"  Imagem: {config.img_size}x{config.img_size} ({config.input_channels} canal)")
    print(f"  Modo de entrada: {config.input_mode}")
    print(f"  Execucoes: {config.n_runs}")
    print(f"  Epocas maximas: {config.epochs}")
    print(f"  Paciencia: {config.patience} epocas")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Medicao de tempo: {'Ativada' if config.measure_time else 'Desativada'}")
    print(f"  Diretorio: {config.experiment_dir}")
    
    print(f"\nBases de teste configuradas:")
    for dataset in config.test_datasets:
        print(f"  - {dataset['name']}:")
        print(f"      Imagens: {dataset['images_dir']}")
        print(f"      Mascaras: {dataset['masks_dir']}")
    
    # Preparar dados de treino
    print("\nCarregando dados de treino...")
    try:
        full_dataset = FundusSegmentationDataset(
            config.train_images_dir,
            config.train_masks_dir,
            img_size=config.img_size,
            input_mode=config.input_mode
        )
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    if len(full_dataset) < 10:
        print(f"Poucos dados: {len(full_dataset)} imagens. Minimo necessario: 10")
        return
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if train_size < 2 or val_size < 2:
        print(f"Dados insuficientes para treino/validacao: Treino={train_size}, Val={val_size}")
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
    
    print(f"  Treino: {len(train_dataset)} imagens ({config.input_mode})")
    print(f"  Validacao: {len(val_dataset)} imagens ({config.input_mode})")
    
    # Multiplos treinamentos
    all_histories = []
    all_best_dices = []
    all_training_times = []
    all_run_summaries = []
    all_metrics_dfs = []
    all_time_dfs = []
    
    for run_id in range(config.n_runs):
        run_start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"# EXECUCAO {run_id+1}/{config.n_runs} - {config.model_name}")
        print(f"{'#'*70}")
        
        try:
            input_channels = 1 if config.input_mode == 'grayscale' else 3
            model = ResNet101UNet(
                num_classes=config.num_classes,
                pretrained=config.pretrained,
                input_channels=input_channels
            )
            model = model.to(config.device)
            
            test_input = torch.randn(1, input_channels, 224, 224).to(config.device)
            test_output = model(test_input)
            print(f"Teste forward pass - Input: {test_input.shape}, Output: {test_output.shape}")
        except Exception as e:
            print(f"Erro ao criar modelo: {e}")
            import traceback
            traceback.print_exc()
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
        
        print(f"\nTempo total da execucao {run_id+1}: {run_total_time:.2f}s")
        
        del model
        torch.cuda.empty_cache()
    
    # Relatorio consolidado de treino
    if all_run_summaries and all_metrics_dfs and all_time_dfs:
        print("\n" + "="*70)
        print("CRIANDO RELATORIO CONSOLIDADO FINAL (TREINO)")
        print("="*70)
        create_consolidated_report(config, all_run_summaries, all_metrics_dfs, all_time_dfs)
    
    if all_best_dices:
        print("\n" + "="*70)
        print("CALCULANDO MEDIAS ENTRE EXECUCOES")
        print("="*70)
        
        best_dice_array = np.array(all_best_dices)
        print(f"\nMelhor Dice por execucao:")
        for i, dice in enumerate(all_best_dices):
            print(f"  Run {i+1}: {dice:.4f}")
        print(f"\nMedia dos melhores Dices: {best_dice_array.mean():.4f} ± {best_dice_array.std():.4f}")
        
        if all_training_times:
            print(f"\nTempo total por execucao:")
            for i, t in enumerate(all_training_times):
                print(f"  Run {i+1}: {t:.2f}s")
            print(f"\nTempo medio por execucao: {np.mean(all_training_times):.2f}s ± {np.std(all_training_times):.2f}s")
    
    # ============================================
    # TESTE EM MULTIPLAS BASES (SEPARADAMENTE)
    # ============================================
    print("\n" + "="*70)
    print("INICIANDO TESTE DOS MODELOS EM MULTIPLAS BASES (SEPARADAS)")
    print("="*70)
    
    all_test_results = test_model_average_detailed(config)
    
    # Relatorio comparativo entre bases
    print("\n" + "="*70)
    print("GERANDO RELATORIO COMPARATIVO ENTRE BASES")
    print("="*70)
    df_cross = create_cross_dataset_report(config, all_test_results)
    
    # Visualizacao final
    print("\n" + "="*70)
    print("VISUALIZANDO RESULTADOS")
    print("="*70)
    visualize_results(config)
    
    print("\n" + "="*70)
    print(f"EXPERIMENTO CONCLUIDO - {config.model_name}")
    print("="*70)
    print(f"\nResultados individuais (treino): {config.experiment_dir}")
    print(f"Modelos salvos: {config.models_dir}")
    print(f"Relatorios consolidados (treino): {config.reports_dir}")
    print(f"Resultados de teste (por base): {config.test_results_dir}")
    print(f"Comparativo entre bases: {os.path.join(config.test_results_dir, 'COMPARATIVO_ENTRE_BASES')}")
    
    if all_training_times:
        print(f"\nRESUMO DE TEMPO:")
        print(f"  Tempo total medio por execucao: {np.mean(all_training_times):.2f}s")
        print(f"  Tempo total de todas as execucoes: {np.sum(all_training_times):.2f}s")

if __name__ == "__main__":
    main()
