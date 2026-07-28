import cv2
import numpy as np
import matplotlib.pyplot as plt 
#from bm3d import bm3d
 
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread

def main():
    # Verifica se o caminho da imagem foi passado como argumento
 
    caminho = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Original/65_A.png"

    try:
        # Lê a imagem (RGB, valores entre 0 e 1)
        img_rgb = imread(caminho)
        # Se a imagem for RGBA (com transparência), descarta o canal alfa
        if img_rgb.ndim == 3 and img_rgb.shape[-1] == 4:
            img_rgb = img_rgb[:, :, :3]
        # Se a imagem estiver em escala de cinza (2D), converte para RGB para padronizar
        elif img_rgb.ndim == 2:
            img_rgb = plt.cm.gray(img_rgb)[:, :, :3]
    except Exception as e:
        print(f"Erro ao ler a imagem: {e}")
        return

    # Extrai os canais R, G, B (matrizes 2D)
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    # Cria a figura com 2x2 subplots e tamanho adequado para artigo
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # --- Plotagem conforme sua estrutura, mas sem SOBEL ---
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Imagem Original', fontsize=12)
    axes[0, 0].axis('off')

    # Canal Vermelho (em tons de cinza - quanto mais branco, maior a intensidade do vermelho)
    axes[0, 1].imshow(R, cmap='gray')
    axes[0, 1].set_title('Canal Vermelho', fontsize=12)
    axes[0, 1].axis('off')

    # Canal Verde (em tons de cinza)
    axes[1, 0].imshow(G, cmap='gray')
    axes[1, 0].set_title('Canal Verde', fontsize=12)
    axes[1, 0].axis('off')

    # Canal Azul (em tons de cinza)
    axes[1, 1].imshow(B, cmap='gray')
    axes[1, 1].set_title('Canal Azul', fontsize=12)
    axes[1, 1].axis('off')

    # Ajusta o espaçamento para deixar as imagens bem próximas (ideal para artigos)
    plt.subplots_adjust(
        wspace=0.05,   # espaçamento horizontal entre colunas
        hspace=0.10,   # espaçamento vertical entre linhas
        left=0.02, 
        right=0.98,
        bottom=0.02, 
        top=0.95
    )

    # Salva a figura com alta resolução e sem bordas extras
    nome_saida = "comparacao_canais_rgb.png"
    fig.savefig(nome_saida, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Figura salva como '{nome_saida}'")

    # Exibe a figura na tela (opcional)
    plt.show()

if __name__ == "__main__":
    main()