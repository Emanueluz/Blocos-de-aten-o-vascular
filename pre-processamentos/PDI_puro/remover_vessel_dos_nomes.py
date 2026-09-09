#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para remover '_vessel' dos nomes das imagens
"""

import os
import re
from glob import glob

# Diretório das máscaras
MASKS_DIR = "/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/Ground truth"

def main():
    print("="*50)
    print("REMOVENDO '_vessel' DOS NOMES DAS IMAGENS")
    print("="*50)
    print(f"Diretório: {MASKS_DIR}")
    print()
    
    # Verificar se o diretório existe
    if not os.path.exists(MASKS_DIR):
        print(f"❌ ERRO: Diretório não encontrado: {MASKS_DIR}")
        return
    
    # Listar todos os arquivos
    files = glob(os.path.join(MASKS_DIR, '*'))
    files = [f for f in files if os.path.isfile(f)]
    
    # Padrões a serem removidos
    patterns = ['_vessel', '_mask', '_gt', '_label']
    
    renamed_count = 0
    renamed_files = []
    
    print("📋 Arquivos que serão renomeados:")
    for file in files:
        filename = os.path.basename(file)
        new_filename = filename
        
        for pattern in patterns:
            new_filename = new_filename.replace(pattern, '')
        
        if filename != new_filename:
            print(f"  📄 {filename} -> {new_filename}")
            renamed_files.append((file, new_filename))
    
    print()
    print(f"⚠️  Serão renomeados {len(renamed_files)} arquivos.")
    
    if len(renamed_files) == 0:
        print("✅ Nenhum arquivo precisa ser renomeado.")
        return
    
    # Perguntar se deseja continuar
    resposta = input("🔄 Deseja continuar com a renomeação? (s/N): ")
    if resposta.lower() != 's':
        print("❌ Operação cancelada.")
        return
    
    print()
    print("🔄 Renomeando arquivos...")
    
    for old_file, new_name in renamed_files:
        new_file = os.path.join(MASKS_DIR, new_name)
        
        if os.path.exists(new_file):
            print(f"⚠️  Arquivo já existe: {new_name} - Pulando")
            continue
        
        os.rename(old_file, new_file)
        print(f"  ✅ {os.path.basename(old_file)} -> {new_name}")
        renamed_count += 1
    
    print()
    print("="*50)
    print(f"✅ Renomeados: {renamed_count} arquivos")
    print("="*50)
    
    # Mostrar exemplo dos arquivos renomeados
    print()
    print("📋 Exemplo de arquivos no diretório:")
    for f in sorted(glob(os.path.join(MASKS_DIR, '*')))[:10]:
        print(f"  - {os.path.basename(f)}")

if __name__ == "__main__":
    main()