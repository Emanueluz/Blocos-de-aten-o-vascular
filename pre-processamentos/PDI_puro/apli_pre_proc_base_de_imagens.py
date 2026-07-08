import cv2
import numpy as np
import os
from pathlib import Path
import time
 
# ============================================================
# CONFIGURAÇÕES
# ============================================================
 
DIRETORIO_ENTRADA = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original"
 
# DIRETÓRIOS DE SAÍDA SEPARADOS (cada um é um diretório raiz)
DIRETORIO_R =          "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/canal_R"
DIRETORIO_G =          "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/canal_G"
DIRETORIO_B =          "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/canal_B"
DIRETORIO_MEDIA =      "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/media"
DIRETORIO_MAX =        "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/max"
DIRETORIO_SOMA =       "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/soma"
DIRETORIO_CANNY =      "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/canny"
DIRETORIO_EQUALIZADO = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/equalizado"

# Extensões de imagem suportadas
EXTENSOES = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================

def sobel_completo_canal(canal):
    """
    Calcula o Sobel em todas as direções para um canal específico
    """
    canal_float = canal.astype(np.float32)
    
    # Sobel Horizontal e Vertical
    sobel_x = cv2.Sobel(canal_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(canal_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # Sobel Diagonal (45° e 135°)
    kernel_45 = np.array([[-2, -1, 0],
                          [-1,  0, 1],
                          [ 0,  1, 2]], dtype=np.float32)
    
    kernel_135 = np.array([[ 0, -1, -2],
                           [ 1,  0, -1],
                           [ 2,  1,  0]], dtype=np.float32)
    
    sobel_45 = cv2.filter2D(canal_float, cv2.CV_32F, kernel_45)
    sobel_135 = cv2.filter2D(canal_float, cv2.CV_32F, kernel_135)
    
    # Magnitude completa
    sobel_completo = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_45**2 + sobel_135**2)
    sobel_completo = cv2.convertScaleAbs(sobel_completo)
    
    return sobel_completo


def processar_imagem(caminho_imagem):
    """
    Processa uma única imagem e salva em diretórios separados
    """
    # Carregar imagem
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"❌ Erro ao carregar: {caminho_imagem}")
        return False
    
    # Redimensionar para 512x512
    img = cv2.resize(img, (512,512))
    
    # Separar canais
    b, g, r = cv2.split(img)
    
    # Processar cada canal
    sobel_r = sobel_completo_canal(r)
    sobel_g = sobel_completo_canal(g)
    sobel_b = sobel_completo_canal(b)
    
    # Método 1: Média dos 3 canais
    sobel_media = cv2.addWeighted(
        cv2.addWeighted(sobel_r, 0.33, sobel_g, 0.33, 0),
        1.0, sobel_b, 0.34, 0
    )
    
    # Método 2: Máximo dos 3 canais
    sobel_max = np.maximum(
        np.maximum(sobel_r, sobel_g),
        sobel_b
    )
    
    # Método 3: Soma normalizada
    sobel_soma = cv2.add(cv2.add(sobel_r, sobel_g), sobel_b)
    sobel_soma = cv2.normalize(sobel_soma, None, 0, 255, cv2.NORM_MINMAX)
    
    # Método 4: Soma suavizada + Canny
    sobel_soma_suavizado = cv2.bilateralFilter(sobel_soma, 9, 90, 90)
    sobel_canny = cv2.Canny(sobel_soma_suavizado, threshold1=30, threshold2=70)
    
    # Método 5: Soma equalizada
    sobel_soma_equalizado = cv2.equalizeHist(sobel_soma)
    
    # Nome base do arquivo
    nome_base = Path(caminho_imagem).stem
    
    # ============================================================
    # CRIAR DIRETÓRIOS SEPARADOS
    # ============================================================
    
    # Criar cada diretório individualmente
    os.makedirs(DIRETORIO_R, exist_ok=True)
    os.makedirs(DIRETORIO_G, exist_ok=True)
    os.makedirs(DIRETORIO_B, exist_ok=True)
    os.makedirs(DIRETORIO_MEDIA, exist_ok=True)
    os.makedirs(DIRETORIO_MAX, exist_ok=True)
    os.makedirs(DIRETORIO_SOMA, exist_ok=True)
    os.makedirs(DIRETORIO_CANNY, exist_ok=True)
    os.makedirs(DIRETORIO_EQUALIZADO, exist_ok=True)
    
    # ============================================================
    # SALVAR EM DIRETÓRIOS SEPARADOS
    # ============================================================
    
    # Salvar canal R
    cv2.imwrite(os.path.join(DIRETORIO_R, f"{nome_base}.png"), sobel_r)
    
    # Salvar canal G
    cv2.imwrite(os.path.join(DIRETORIO_G, f"{nome_base}.png"), sobel_g)
    
    # Salvar canal B
    cv2.imwrite(os.path.join(DIRETORIO_B, f"{nome_base}.png"), sobel_b)
    
    # Salvar média
    cv2.imwrite(os.path.join(DIRETORIO_MEDIA, f"{nome_base}.png"), sobel_media)
    
    # Salvar máximo
    cv2.imwrite(os.path.join(DIRETORIO_MAX, f"{nome_base}.png"), sobel_max)
    
    # Salvar soma
    cv2.imwrite(os.path.join(DIRETORIO_SOMA, f"{nome_base}.png"), sobel_soma)
    
    # Salvar Canny
    cv2.imwrite(os.path.join(DIRETORIO_CANNY, f"{nome_base}.png"), sobel_canny)
    
    # Salvar equalizado
    cv2.imwrite(os.path.join(DIRETORIO_EQUALIZADO, f"{nome_base}.png"), sobel_soma_equalizado)
    
    return True


def processar_diretorio(diretorio_entrada):
    """
    Processa todas as imagens em um diretório
    """
    # Verificar se diretório de entrada existe
    if not os.path.exists(diretorio_entrada):
        print(f"❌ Diretório de entrada não encontrado: {diretorio_entrada}")
        return
    
    # Listar arquivos de imagem
    imagens = []
    for ext in EXTENSOES:
        imagens.extend(Path(diretorio_entrada).glob(f"*{ext}"))
        imagens.extend(Path(diretorio_entrada).glob(f"*{ext.upper()}"))
    
    if not imagens:
        print(f"❌ Nenhuma imagem encontrada em: {diretorio_entrada}")
        return
    
    print(f"\n{'='*60}")
    print(f"📁 Processando {len(imagens)} imagens...")
    print(f"📂 Entrada: {diretorio_entrada}")
    print(f"{'='*60}\n")
    
    print("📂 Diretórios de saída:")
    print(f"   ├── {DIRETORIO_R}")
    print(f"   ├── {DIRETORIO_G}")
    print(f"   ├── {DIRETORIO_B}")
    print(f"   ├── {DIRETORIO_MEDIA}")
    print(f"   ├── {DIRETORIO_MAX}")
    print(f"   ├── {DIRETORIO_SOMA}")
    print(f"   ├── {DIRETORIO_CANNY}")
    print(f"   └── {DIRETORIO_EQUALIZADO}")
    print()
    
    # Processar cada imagem
    sucessos = 0
    falhas = 0
    inicio = time.time()
    
    for i, img_path in enumerate(imagens, 1):
        print(f"[{i}/{len(imagens)}] Processando: {img_path.name}")
        
        if processar_imagem(str(img_path)):
            sucessos += 1
        else:
            falhas += 1
    
    # Estatísticas finais
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
    
    print(f"\n📁 Resultados salvos em diretórios separados:")
    print(f"   - Canal R: {DIRETORIO_R}")
    print(f"   - Canal G: {DIRETORIO_G}")
    print(f"   - Canal B: {DIRETORIO_B}")
    print(f"   - Média: {DIRETORIO_MEDIA}")
    print(f"   - Máximo: {DIRETORIO_MAX}")
    print(f"   - Soma: {DIRETORIO_SOMA}")
    print(f"   - Canny: {DIRETORIO_CANNY}")
    print(f"   - Equalizado: {DIRETORIO_EQUALIZADO}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Processar diretório
    processar_diretorio(DIRETORIO_ENTRADA)
    
    print("\n🎯 Processamento completo!")