import cv2
import os
import numpy as np
from pathlib import Path

def preprocessar_imagens(diretorio_entrada, diretorio_saida, extensoes=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
 
    
    # Cria o diretório de saída se não existir
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    
    # Lista todos os arquivos no diretório de entrada
    arquivos = os.listdir(diretorio_entrada)
    
    # Filtra apenas os arquivos com extensões de imagem
    imagens = [f for f in arquivos if f.lower().endswith(extensoes)]
    
    if not imagens:
        print(f"Nenhuma imagem encontrada em {diretorio_entrada}")
        return
    
    print(f"Processando {len(imagens)} imagens...")
    
    for nome_arquivo in imagens:
        try:
            caminho_entrada = os.path.join(diretorio_entrada, nome_arquivo)
            
            # Lê a imagem em escala de cinza
            img = cv2.imread(caminho_entrada, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Erro ao ler: {nome_arquivo}")
                continue
            
            # Redimensiona para 224x224
            img = cv2.resize(img, (224, 224))
            
            # Binariza: pixels > 0 viram 255
            _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
            
            # Monta o caminho de saída
            nome_saida = f"processed_{nome_arquivo}"
            caminho_saida = os.path.join(diretorio_saida, nome_saida)
            
            # Salva a imagem processada
            cv2.imwrite(caminho_saida, img)
            
            print(f"✓ {nome_arquivo} -> {nome_saida}")
            
        except Exception as e:
            print(f"✗ Erro ao processar {nome_arquivo}: {e}")
    
    print(f"\nProcessamento concluído! Imagens salvas em: {diretorio_saida}")

entrada= '/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg/annotation/'
saida='/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg/annotation_b'
preprocessar_imagens(entrada,saida)