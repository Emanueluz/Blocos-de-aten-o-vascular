import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def merge_images_horizontally(dir1, dir2, output_dir, resize_width=None, resize_height=None, 
                              add_spacing=0, background_color=(0, 0, 0), 
                              show_differences=True):
    """
    Junta imagens de dois diretórios horizontalmente (lado a lado)
    baseado no nome do arquivo.
    
    Args:
        show_differences: Se True, mostra quais arquivos estão em um diretório mas não no outro
    """
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Obter todos os arquivos de imagem dos diretórios
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    files1 = []
    files2 = []
    
    print("📂 Procurando imagens...")
    
    for ext in extensions:
        files1.extend(glob(os.path.join(dir1, ext)))
        files2.extend(glob(os.path.join(dir2, ext)))
    
    # Extrair nomes dos arquivos (sem extensão para comparação)
    names1_full = {os.path.basename(f).lower(): f for f in files1}
    names2_full = {os.path.basename(f).lower(): f for f in files2}
    
    # Nomes sem extensão para comparação mais flexível
    names1_no_ext = {}
    for name, path in names1_full.items():
        base_name = os.path.splitext(name)[0]
        if base_name not in names1_no_ext:
            names1_no_ext[base_name] = []
        names1_no_ext[base_name].append(name)
    
    names2_no_ext = {}
    for name, path in names2_full.items():
        base_name = os.path.splitext(name)[0]
        if base_name not in names2_no_ext:
            names2_no_ext[base_name] = []
        names2_no_ext[base_name].append(name)
    
    # Mostrar diferenças
    if show_differences:
        print("\n" + "="*60)
        print("📊 ANÁLISE DE NOMES DOS ARQUIVOS")
        print("="*60)
        
        # Arquivos apenas no diretório 1
        only_dir1 = set(names1_no_ext.keys()) - set(names2_no_ext.keys())
        if only_dir1:
            print(f"\n📁 Arquivos APENAS no Diretório 1 ({len(only_dir1)}):")
            for name in sorted(only_dir1)[:20]:  # Mostra até 20
                ext = names1_full.get(name + '.png') or names1_full.get(name + '.jpg') or ''
                print(f"  - {name}{ext if ext else ''}")
            if len(only_dir1) > 20:
                print(f"  ... e mais {len(only_dir1) - 20} arquivos")
        
        # Arquivos apenas no diretório 2
        only_dir2 = set(names2_no_ext.keys()) - set(names1_no_ext.keys())
        if only_dir2:
            print(f"\n📁 Arquivos APENAS no Diretório 2 ({len(only_dir2)}):")
            for name in sorted(only_dir2)[:20]:
                ext = names2_full.get(name + '.png') or names2_full.get(name + '.jpg') or ''
                print(f"  - {name}{ext if ext else ''}")
            if len(only_dir2) > 20:
                print(f"  ... e mais {len(only_dir2) - 20} arquivos")
        
        # Arquivos com extensões diferentes
        common_names = set(names1_no_ext.keys()) & set(names2_no_ext.keys())
        diff_extensions = []
        for name in common_names:
            ext1 = set()
            ext2 = set()
            for f in names1_no_ext[name]:
                ext1.add(os.path.splitext(f)[1])
            for f in names2_no_ext[name]:
                ext2.add(os.path.splitext(f)[1])
            
            if ext1 != ext2:
                diff_extensions.append((name, ext1, ext2))
        
        if diff_extensions:
            print(f"\n⚠️ Arquivos com EXTENSÕES DIFERENTES ({len(diff_extensions)}):")
            for name, ext1, ext2 in sorted(diff_extensions)[:20]:
                print(f"  - {name}: Dir1={ext1}, Dir2={ext2}")
            if len(diff_extensions) > 20:
                print(f"  ... e mais {len(diff_extensions) - 20} arquivos")
        
        # Resumo
        print("\n" + "-"*60)
        print("📊 RESUMO:")
        print(f"  Total arquivos Dir1: {len(names1_full)}")
        print(f"  Total arquivos Dir2: {len(names2_full)}")
        print(f"  Nomes comuns (exata): {len(set(names1_full.keys()) & set(names2_full.keys()))}")
        print(f"  Nomes comuns (sem extensão): {len(common_names)}")
        print(f"  Apenas no Dir1: {len(only_dir1)}")
        print(f"  Apenas no Dir2: {len(only_dir2)}")
        print(f"  Extensões diferentes: {len(diff_extensions)}")
        print("="*60 + "\n")
    
    # Encontrar nomes comuns (considerando sem extensão)
    # Para cada nome comum, escolher o primeiro arquivo de cada diretório
    common_pairs = []
    for base_name in sorted(set(names1_no_ext.keys()) & set(names2_no_ext.keys())):
        # Pegar o primeiro arquivo de cada diretório
        file1 = names1_full.get(names1_no_ext[base_name][0])
        file2 = names2_full.get(names2_no_ext[base_name][0])
        if file1 and file2:
            common_pairs.append((base_name, file1, file2))
    
    if not common_pairs:
        print(f"❌ Nenhum arquivo com nome comum encontrado!")
        print(f"   Dir1: {len(names1_full)} arquivos")
        print(f"   Dir2: {len(names2_full)} arquivos")
        return
    
    print(f"✅ Encontrados {len(common_pairs)} pares de imagens para mesclar")
    
    # Processar cada par de imagens
    successful = 0
    failed = 0
    failed_files = []
    
    for base_name, path1, path2 in tqdm(common_pairs, desc="🔄 Mesclando"):
        try:
            # Obter nome do arquivo para salvar (usar nome do dir1 ou dir2)
            name1 = os.path.basename(path1)
            name2 = os.path.basename(path2)
            
            # Usar o nome que existe em ambos ou o mais longo
            if name1 in names2_full:
                save_name = name1
            elif name2 in names1_full:
                save_name = name2
            else:
                # Se nenhum existe exatamente, usar o nome com extensão do dir1
                save_name = name1
            
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None:
                print(f"⚠️ Não foi possível carregar: {path1}")
                failed += 1
                failed_files.append((base_name, "Erro ao carregar imagem 1"))
                continue
                
            if img2 is None:
                print(f"⚠️ Não foi possível carregar: {path2}")
                failed += 1
                failed_files.append((base_name, "Erro ao carregar imagem 2"))
                continue
            
            # Redimensionar se especificado
            if resize_width is not None and resize_height is not None:
                img1 = cv2.resize(img1, (resize_width, resize_height))
                img2 = cv2.resize(img2, (resize_width, resize_height))
            else:
                # Garantir mesma altura
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                if h1 != h2:
                    max_height = max(h1, h2)
                    if h1 < max_height:
                        new_w1 = int(w1 * (max_height / h1))
                        img1 = cv2.resize(img1, (new_w1, max_height))
                    if h2 < max_height:
                        new_w2 = int(w2 * (max_height / h2))
                        img2 = cv2.resize(img2, (new_w2, max_height))
            
            # Juntar horizontalmente
            if add_spacing > 0:
                h_final = max(img1.shape[0], img2.shape[0])
                w_final = img1.shape[1] + img2.shape[1] + add_spacing
                merged = np.full((h_final, w_final, 3), background_color, dtype=np.uint8)
                merged[0:img1.shape[0], 0:img1.shape[1]] = img1
                merged[0:img2.shape[0], img1.shape[1] + add_spacing:] = img2
            else:
                merged = np.hstack([img1, img2])
            
            # Salvar
            output_path = os.path.join(output_dir, save_name)
            cv2.imwrite(output_path, merged)
            successful += 1
            
        except Exception as e:
            print(f"❌ Erro ao processar {base_name}: {str(e)}")
            failed += 1
            failed_files.append((base_name, str(e)))
    
    # Relatório final
    print("\n" + "="*60)
    print("📊 RELATÓRIO FINAL")
    print("="*60)
    print(f"✅ Imagens processadas com sucesso: {successful}")
    print(f"❌ Falhas: {failed}")
    print(f"📁 Total de imagens geradas: {successful}")
    print(f"📂 Diretório de saída: {output_dir}")
    
    if failed_files:
        print("\n⚠️ Arquivos com falha:")
        for name, error in failed_files[:10]:
            print(f"  - {name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... e mais {len(failed_files) - 10} arquivos")
    
    print("="*60)

# ============================================
# MAIN COM DIRETÓRIOS DEFINIDOS
# ============================================

def main():
    """
    Função principal com diretórios definidos diretamente aqui
    """
    print("\n" + "="*60)
    print("🖼️  JUNTAR IMAGENS HORIZONTALMENTE")
    print("="*60)
    
    # ============================================
    # DEFINA SEUS DIRETÓRIOS AQUI
    # ============================================
    
    # Diretórios de entrada
    DIR_ESQUERDA = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Original"
    DIR_DIREITA = "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/test/soma"
    
    # Diretório de saída
    DIR_SAIDA = "./imagens_mergeadas_test"
    
    # Configurações
    LARGURA = 512 
    ALTURA = 512
    ESPACAMENTO = 0  # pixels entre as imagens
    MOSTRAR_DIFERENCAS = True  # Mostrar quais nomes são diferentes
    
    # ============================================
    
    # Mostrar configurações
    print(f"\n📂 Diretório ESQUERDA: {DIR_ESQUERDA}")
    print(f"📂 Diretório DIREITA: {DIR_DIREITA}")
    print(f"📁 Diretório SAÍDA: {DIR_SAIDA}")
    print(f"📐 Redimensionar para: {LARGURA}x{ALTURA}")
    print(f"📏 Espaçamento: {ESPACAMENTO}px")
    print(f"🔍 Mostrar diferenças: {'Sim' if MOSTRAR_DIFERENCAS else 'Não'}")
    
    # Verificar se os diretórios existem
    if not os.path.exists(DIR_ESQUERDA):
        print(f"\n❌ ERRO: Diretório não encontrado: {DIR_ESQUERDA}")
        return
    
    if not os.path.exists(DIR_DIREITA):
        print(f"\n❌ ERRO: Diretório não encontrado: {DIR_DIREITA}")
        return
    
    # Executar mesclagem
    print("\n🔄 Processando...")
    merge_images_horizontally(
        DIR_ESQUERDA,
        DIR_DIREITA,
        DIR_SAIDA,
        resize_width=LARGURA,
        resize_height=ALTURA,
        add_spacing=ESPACAMENTO,
        show_differences=MOSTRAR_DIFERENCAS
    )
    
    print("\n✅ Concluído!")

# ============================================
# EXECUTAR
# ============================================

if __name__ == "__main__":
    main()
    # Diretórios de entrada
    