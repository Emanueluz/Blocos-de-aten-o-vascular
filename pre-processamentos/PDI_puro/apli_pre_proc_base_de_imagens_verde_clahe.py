import cv2
import os
from pathlib import Path
import time

# ============================================================
# CONFIGURAÇÕES
# ============================================================

DIRETORIO_ENTRADA = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original"
DIRETORIO_SAIDA = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/canal_G_clahe"

# Parâmetros do CLAHE (ajuste conforme necessidade)
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)

# Extensões suportadas
EXTENSOES = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# ============================================================
# FUNÇÃO PARA APLICAR CLAHE NO CANAL VERDE
# ============================================================

def processar_imagem(caminho_imagem):
    # Carrega a imagem colorida
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"❌ Erro ao carregar: {caminho_imagem}")
        return False

    # Redimensiona para 512x512 (opcional, mantido do original)
    img = cv2.resize(img, (512, 512))

    # Extrai o canal verde (índice 1 em BGR)
    canal_verde = img[:, :, 1]

    # Aplica CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    canal_verde_clahe = clahe.apply(canal_verde)

    # Cria diretório de saída se não existir
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

    # Salva a imagem processada
    nome_base = Path(caminho_imagem).stem
    caminho_saida = os.path.join(DIRETORIO_SAIDA, f"{nome_base}.png")
    cv2.imwrite(caminho_saida, canal_verde_clahe)

    return True

# ============================================================
# PROCESSAMENTO EM LOTE
# ============================================================

def processar_diretorio():
    if not os.path.exists(DIRETORIO_ENTRADA):
        print(f"❌ Diretório de entrada não encontrado: {DIRETORIO_ENTRADA}")
        return

    # Lista todos os arquivos de imagem
    imagens = []
    for ext in EXTENSOES:
        imagens.extend(Path(DIRETORIO_ENTRADA).glob(f"*{ext}"))
        imagens.extend(Path(DIRETORIO_ENTRADA).glob(f"*{ext.upper()}"))

    if not imagens:
        print(f"❌ Nenhuma imagem encontrada em: {DIRETORIO_ENTRADA}")
        return

    print(f"\n{'='*60}")
    print(f"📁 Processando {len(imagens)} imagens...")
    print(f"📂 Entrada: {DIRETORIO_ENTRADA}")
    print(f"📂 Saída: {DIRETORIO_SAIDA}")
    print(f"🔧 CLAHE: clipLimit={CLIP_LIMIT}, tileGridSize={TILE_GRID_SIZE}")
    print(f"{'='*60}\n")

    sucessos = 0
    falhas = 0
    inicio = time.time()

    for i, img_path in enumerate(imagens, 1):
        print(f"[{i}/{len(imagens)}] Processando: {img_path.name}")
        if processar_imagem(str(img_path)):
            sucessos += 1
        else:
            falhas += 1

    tempo_total = time.time() - inicio
    print(f"\n{'='*60}")
    print(f"✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"📊 Estatísticas:")
    print(f"   - Total: {len(imagens)}")
    print(f"   - Sucessos: {sucessos}")
    print(f"   - Falhas: {falhas}")
    print(f"   - Tempo total: {tempo_total:.2f} s")
    print(f"   - Média por imagem: {tempo_total/len(imagens):.2f} s")
    print(f"{'='*60}")

# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    processar_diretorio()
    print("\n🎯 Processamento completo!")