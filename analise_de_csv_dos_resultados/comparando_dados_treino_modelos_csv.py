import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def calcular_medias_arquivo(arquivo_csv):
    """
    Calcula as médias de TODAS as colunas numéricas de um único arquivo CSV.
    
    Args:
        arquivo_csv (str): Caminho do arquivo CSV
    
    Returns:
        dict: Dicionário com as médias calculadas
    """
    # Ler o CSV
    df = pd.read_csv(arquivo_csv)
    
    print(f"  📋 Colunas encontradas: {list(df.columns)}")
    
    # Identificar TODAS as colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calcular as médias para TODAS as colunas numéricas
    medias = {}
    for col in colunas_numericas:
        medias[col] = df[col].mean()
    
    # Adicionar informações gerais (não numéricas)
    medias['arquivo_origem'] = os.path.basename(arquivo_csv)
    medias['n_execucoes'] = len(df)
    
    # Adicionar valores não numéricos que podem ser úteis
    for col in ['model_name', 'img_size', 'input_channels']:
        if col in df.columns:
            # Se for numérico, já foi calculado a média
            if col not in medias:
                medias[col] = df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 'desconhecido'
    
    # Calcular desvio padrão para TODAS as colunas numéricas
    for col in colunas_numericas:
        medias[f'{col}_std'] = df[col].std()
        medias[f'{col}_min'] = df[col].min()
        medias[f'{col}_max'] = df[col].max()
        medias[f'{col}_mediana'] = df[col].median()
    
    return medias

def processar_multiplos_csvs(lista_csvs, arquivo_saida):
    """
    Processa múltiplos arquivos CSV e gera um CSV com as médias de cada um.
    
    Args:
        lista_csvs (list): Lista com os caminhos dos arquivos CSV
        arquivo_saida (str): Caminho do arquivo CSV de saída
    """
    print("="*70)
    print("📊 PROCESSANDO MÚLTIPLOS ARQUIVOS CSV")
    print("="*70)
    
    todas_medias = []
    todas_colunas = set()
    
    # Primeiro, descobrir todas as colunas disponíveis
    for arquivo in lista_csvs:
        if os.path.exists(arquivo):
            df = pd.read_csv(arquivo)
            todas_colunas.update(df.columns)
    
    print(f"\n📋 Colunas encontradas nos arquivos: {len(todas_colunas)}")
    
    for i, arquivo in enumerate(lista_csvs, 1):
        print(f"\n📂 Processando arquivo {i}/{len(lista_csvs)}: {os.path.basename(arquivo)}")
        
        if not os.path.exists(arquivo):
            print(f"  ⚠️ Arquivo não encontrado: {arquivo}")
            continue
        
        try:
            medias = calcular_medias_arquivo(arquivo)
            medias['indice'] = i
            todas_medias.append(medias)
            print(f"  ✅ Processado com sucesso ({medias['n_execucoes']} execuções)")
            print(f"  📊 {len(medias)} métricas calculadas")
        except Exception as e:
            print(f"  ❌ Erro ao processar: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not todas_medias:
        print("\n❌ Nenhum arquivo foi processado com sucesso!")
        return None
    
    # Criar DataFrame com os resultados
    df_resultado = pd.DataFrame(todas_medias)
    
    # Organizar colunas: primeiro as informações gerais, depois as médias, depois os desvios
    colunas_info = ['indice', 'arquivo_origem', 'n_execucoes']
    colunas_modelo = ['model_name', 'img_size', 'input_channels']
    
    # Separar colunas por tipo
    colunas_medias = []
    colunas_std = []
    colunas_min = []
    colunas_max = []
    colunas_mediana = []
    colunas_outras = []
    
    for col in df_resultado.columns:
        if col in colunas_info or col in colunas_modelo:
            continue
        if col.endswith('_std'):
            colunas_std.append(col)
        elif col.endswith('_min'):
            colunas_min.append(col)
        elif col.endswith('_max'):
            colunas_max.append(col)
        elif col.endswith('_mediana'):
            colunas_mediana.append(col)
        else:
            colunas_medias.append(col)
    
    # Ordenar as colunas
    colunas_medias.sort()
    colunas_std.sort()
    colunas_min.sort()
    colunas_max.sort()
    colunas_mediana.sort()
    
    # Montar ordem final
    colunas_finais = colunas_info + colunas_modelo + colunas_medias + colunas_std + colunas_min + colunas_max + colunas_mediana
    colunas_finais = [col for col in colunas_finais if col in df_resultado.columns]
    
    df_resultado = df_resultado[colunas_finais]
    
    # Salvar resultado
    df_resultado.to_csv(arquivo_saida, index=False)
    print(f"\n✅ Resultado salvo em: {arquivo_saida}")
    print(f"📊 Total de colunas: {len(df_resultado.columns)}")
    print(f"📊 Total de linhas: {len(df_resultado)}")
    
    # Exibir resumo
    print("\n" + "="*70)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*70)
    print(f"\nTotal de arquivos processados: {len(todas_medias)}")
    print(f"Total de execuções analisadas: {df_resultado['n_execucoes'].sum()}")
    
    # Mostrar colunas disponíveis
    print(f"\n📋 Colunas disponíveis no arquivo gerado:")
    print("-" * 70)
    for i, col in enumerate(df_resultado.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Exibir tabela comparativa com as principais métricas
    print("\n" + "="*70)
    print("📊 TABELA COMPARATIVA (Métricas Principais)")
    print("="*70)
    
    # Identificar colunas principais
    colunas_principais = ['indice', 'arquivo_origem', 'model_name']
    for col in ['best_val_dice', 'final_val_dice', 'final_val_iou', 
                'final_val_accuracy', 'final_val_sensitivity', 'final_val_specificity']:
        if col in df_resultado.columns:
            colunas_principais.append(col)
    
    df_resumo = df_resultado[colunas_principais]
    print(df_resumo.to_string(index=False))
    
    return df_resultado

def main():
    print("="*70)
    print("📊 PROCESSADOR DE MÚLTIPLOS CSVs")
    print("   Calcula TODAS as métricas de cada arquivo e gera um CSV consolidado")
    print("="*70)
    
    # ================================================================
    # COLOQUE AQUI OS CAMINHOS DOS 6 CSVs QUE VOCÊ QUER PROCESSAR
    # ================================================================
    arquivos_csv = [
        "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza/EfficientNetB0_UNet_cinza/dados_combinados_runs.csv",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza/MobileNetV2_UNet_cinza/dados_combinados_runs.csv",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza/ResNet101_UNet_cinza/dados_combinados_runs.csv",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza/SwinUNet_cinza/dados_combinados_runs.csv",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza/VGG19_UNet_cinza/dados_combinados_runs.csv",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza/ViTUNet_cinza/dados_combinados_runs.csv"
    ]
    
    # Ou use glob para encontrar automaticamente
    # arquivos_csv = glob.glob("/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/*/RELATORIO_DAS_EXECUCOES/RESUMO_EXECUCOES_*.csv")
    
    # Verificar quantos arquivos existem
    arquivos_existentes = [f for f in arquivos_csv if os.path.exists(f)]
    
    print(f"\n📂 Arquivos encontrados: {len(arquivos_existentes)}/{len(arquivos_csv)}")
    
    if len(arquivos_existentes) == 0:
        print("\n❌ Nenhum arquivo encontrado!")
        print("\n📝 Por favor, edite a lista 'arquivos_csv' no código com os caminhos corretos.")
        print("   Ou use glob para encontrar automaticamente.")
        return
    
    # Mostrar arquivos encontrados
    print("\n📋 Arquivos a serem processados:")
    for i, arquivo in enumerate(arquivos_existentes, 1):
        print(f"  {i}. {os.path.basename(arquivo)}")
    
    # Mostrar colunas de exemplo do primeiro arquivo
    print("\n📋 Exemplo de colunas do primeiro arquivo:")
    df_exemplo = pd.read_csv(arquivos_existentes[0])
    print(f"  Total de colunas: {len(df_exemplo.columns)}")
    print(f"  Colunas: {list(df_exemplo.columns)}")
 
    print("   (Pressione ENTER para nome automático)")
    
    
    
    nome_saida ='/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/medias_dados_treino_IMG-CINZA/media_dados_treino_dos_modelos-CINZA.csv'
    
    if nome_saida == "":
        nome_saida = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        if not nome_saida.endswith('.csv'):
            nome_saida += '.csv'
    
    # Processar
    df_resultado = processar_multiplos_csvs(arquivos_existentes, nome_saida)
    
    if df_resultado is not None:
        print("\n" + "="*70)
        print("✅ PROCESSAMENTO CONCLUÍDO")
        print("="*70)
        print(f"\n📁 Arquivo gerado: {nome_saida}")
        print(f"📊 Total de linhas: {len(df_resultado)}")
        print(f"📋 Total de colunas: {len(df_resultado.columns)}")
        
        # Salvar também uma versão transposta para melhor visualização
        df_transposto = df_resultado.set_index('arquivo_origem').T
        nome_transposto = f"{nome_saida}"
        df_transposto.to_csv(nome_transposto)
        print(f"\n📁 Versão transposta salva em: {nome_transposto}")
        
        print("\n📊 Estatísticas das colunas:")
        print("-" * 70)
        colunas_numericas = df_resultado.select_dtypes(include=[np.number]).columns
        print(f"  Colunas numéricas: {len(colunas_numericas)}")
        print(f"  Colunas não numéricas: {len(df_resultado.columns) - len(colunas_numericas)}")
        
        # Mostrar exemplo dos dados
        print("\n📋 Primeiras linhas do arquivo gerado:")
        print(df_resultado.head(3).to_string())

# ============================================
# EXECUTAR
# ============================================

if __name__ == "__main__":
    main() 
    
    