import cv2
import numpy as np
import os
from pathlib import Path
import time

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Parâmetros do CLAHE (mantidos do código original)
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
IMG_SIZE = 224  # Tamanho padrão dos modelos

# Extensões suportadas
EXTENSOES = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# ============================================================
# DIRETÓRIOS DE ENTRADA E SAÍDA
# ============================================================

# --- DIRETÓRIO DE ENTRADA (APENAS UM) ---
INPUT_DIR = "/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/Original"

# --- DIRETÓRIO DE SAÍDA ---
OUTPUT_BASE = "/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/PDI_puro"

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================

def aplicar_clahe(canal):
    """Aplica CLAHE em um canal"""
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    return clahe.apply(canal)

def processar_imagem(caminho_imagem, diretorios_saida):
    """
    Processa uma imagem gerando apenas:
    1. Tons de cinza (gray)
    2. Canal Verde (G)
    3. Verde melhorado (gray_enhanced)
    4. Verde + CLAHE (G_clahe)
    """
    # Carregar imagem
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"❌ Erro ao carregar: {caminho_imagem}")
        return False
    
    # Redimensionar para 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Separar canais BGR
    b, g, r = cv2.split(img)
    
    # ============================================================
    # 1. TONS DE CINZA (GRAY)
    # ============================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ============================================================
    # 2. CANAL VERDE (G)
    # ============================================================
    canal_verde = g
    
    # ============================================================
    # 3. VERDE MELHORADO (GRAY_ENHANCED)
    # Combinação: 0.6*G + 0.4*(0.3*R + 0.1*B)
    # ============================================================
    rb_combined = cv2.addWeighted(r, 0.3, b, 0.1, 0)
    gray_enhanced = cv2.addWeighted(g, 0.6, rb_combined, 0.4, 0)
    
    # ============================================================
    # 4. VERDE + CLAHE (G_CLaHE)
    # ============================================================
    g_clahe = aplicar_clahe(g)
    
    # ============================================================
    # Nome base da imagem
    # ============================================================
    nome_base = Path(caminho_imagem).stem
    
    # ============================================================
    # CRIAR DIRETÓRIOS
    # ============================================================
    os.makedirs(diretorios_saida['gray'], exist_ok=True)
    os.makedirs(diretorios_saida['canal_g'], exist_ok=True)
    os.makedirs(diretorios_saida['gray_enhanced'], exist_ok=True)
    os.makedirs(diretorios_saida['g_clahe'], exist_ok=True)
    
    # ============================================================
    # SALVAR IMAGENS
    # ============================================================
    cv2.imwrite(os.path.join(diretorios_saida['gray'], f"{nome_base}.png"), gray)
    cv2.imwrite(os.path.join(diretorios_saida['canal_g'], f"{nome_base}.png"), canal_verde)
    cv2.imwrite(os.path.join(diretorios_saida['gray_enhanced'], f"{nome_base}.png"), gray_enhanced)
    cv2.imwrite(os.path.join(diretorios_saida['g_clahe'], f"{nome_base}.png"), g_clahe)
    
    return True

def listar_imagens(diretorio):
    """Lista todas as imagens em um diretório"""
    imagens = []
    for ext in EXTENSOES:
        imagens.extend(Path(diretorio).glob(f"*{ext}"))
        imagens.extend(Path(diretorio).glob(f"*{ext.upper()}"))
    return imagens

# ============================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================

def main():
    print("="*60)
    print("🚀 PROCESSAMENTO DE IMAGENS PARA SEGMENTAÇÃO")
    print("="*60)
    print(f"📂 Entrada: {INPUT_DIR}")
    print(f"📂 Saída base: {OUTPUT_BASE}")
    print(f"📐 Tamanho: {IMG_SIZE}x{IMG_SIZE}")
    print(f"🔧 CLAHE: clipLimit={CLIP_LIMIT}, tileGridSize={TILE_GRID_SIZE}")
    print("="*60)
    
    # Verificar diretório de entrada
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Diretório de entrada não encontrado: {INPUT_DIR}")
        return
    
    # Definir diretórios de saída
    diretorios_saida = {
        'gray': os.path.join(OUTPUT_BASE, 'gray'),
        'canal_g': os.path.join(OUTPUT_BASE, 'canal_G'),
        'gray_enhanced': os.path.join(OUTPUT_BASE, 'gray_enhanced'),
        'g_clahe': os.path.join(OUTPUT_BASE, 'G_clahe'),
    }
    
    # Listar imagens
    imagens = listar_imagens(INPUT_DIR)
    
    if not imagens:
        print(f"❌ Nenhuma imagem encontrada em: {INPUT_DIR}")
        return
    
    # ============================================================
    # PROCESSAR IMAGENS
    # ============================================================
    
    print(f"\n📁 Processando {len(imagens)} imagens...")
    print(f"{'='*60}\n")
    
    print("📂 Diretórios de saída:")
    print(f"   ├── {diretorios_saida['gray']}        (Tons de cinza)")
    print(f"   ├── {diretorios_saida['canal_g']}     (Canal Verde)")
    print(f"   ├── {diretorios_saida['gray_enhanced']} (Verde melhorado)")
    print(f"   └── {diretorios_saida['g_clahe']}    (Verde + CLAHE)")
    print()
    
    # ============================================================
    # PROCESSAR IMAGENS
    # ============================================================
    
    sucessos = 0
    falhas = 0
    inicio = time.time()
    
    for i, img_path in enumerate(imagens, 1):
        print(f"[{i}/{len(imagens)}] Processando: {img_path.name}")
        
        if processar_imagem(str(img_path), diretorios_saida):
            sucessos += 1
        else:
            falhas += 1
    
    # ============================================================
    # ESTATÍSTICAS FINAIS
    # ============================================================
    
    tempo_total = time.time() - inicio
    print(f"\n{'='*60}")
    print(f"✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"📊 Estatísticas:")
    print(f"   - Total de imagens: {len(imagens)}")
    print(f"   - Processadas com sucesso: {sucessos}")
    print(f"   - Falhas: {falhas}")
    print(f"   - Tempo total: {tempo_total:.2f} segundos")
    print(f"   - Média por imagem: {tempo_total/len(imagens):.2f} segundos")
    print(f"{'='*60}")
    
    print(f"\n📁 Resultados salvos em: {OUTPUT_BASE}")
    print(f"   - Tons de cinza: {diretorios_saida['gray']}")
    print(f"   - Canal Verde: {diretorios_saida['canal_g']}")
    print(f"   - Verde melhorado: {diretorios_saida['gray_enhanced']}")
    print(f"   - Verde + CLAHE: {diretorios_saida['g_clahe']}")
    print("\n🎯 Processamento completo!")

# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()