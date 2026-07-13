import os
import pandas as pd
import numpy as np
from pathlib import Path

def analisar_runs(diretorio_base):
    """
    Analisa os arquivos run_summary.csv nos subdiretórios run_0 a run_4
    
    Args:
        diretorio_base: Caminho para o diretório principal
    
    Returns:
        DataFrame com estatísticas consolidadas
    """
    
    # Converter para Path object para melhor manipulação
    base_path = Path(diretorio_base)
    
    # Verificar se o diretório existe
    if not base_path.exists():
        print(f"Erro: Diretório {diretorio_base} não encontrado!")
        return None
    
    # Lista para armazenar os DataFrames
    dfs = []
    runs_encontradas = []
    
    # Procurar diretórios run_0 a run_4
    for i in range(5):
        run_dir = base_path / f"run_{i}"
        csv_file = run_dir / "run_summary.csv"
        
        if csv_file.exists():
            try:
                # Ler o CSV
                df = pd.read_csv(csv_file)
                # Adicionar coluna identificando a run
                df['run_number'] = i
                dfs.append(df)
                runs_encontradas.append(i)
                print(f"✓ Run {i} encontrada e carregada")
            except Exception as e:
                print(f"✗ Erro ao ler run_{i}: {e}")
        else:
            print(f"✗ Run {i} não encontrada (arquivo {csv_file} não existe)")
    
    if not dfs:
        print("Nenhum arquivo run_summary.csv encontrado!")
        return None
    
    # Concatenar todos os DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal de runs analisadas: {len(dfs)}")
    print(f"Runs encontradas: {runs_encontradas}")
    
    # Selecionar colunas para análise
    colunas_interesse = [
        'best_val_dice', 'best_epoch', 'total_epochs', 
        'total_training_time', 'final_val_dice', 'final_val_loss',
        'final_val_accuracy', 'final_val_iou', 
        'final_val_sensitivity', 'final_val_specificity'
    ]
    
    # Filtrar apenas colunas que existem
    colunas_existentes = [col for col in colunas_interesse if col in df_combined.columns]
    
    # Calcular estatísticas
    estatisticas = {}
    
    for coluna in colunas_existentes:
        # Converter para numérico (tratar possíveis erros)
        df_combined[coluna] = pd.to_numeric(df_combined[coluna], errors='coerce')
        
        # Remover NaN
        valores = df_combined[coluna].dropna()
        
        if len(valores) > 0:
            estatisticas[coluna] = {
                'media': valores.mean(),
                'desvio_padrao': valores.std(),
                'min': valores.min(),
                'max': valores.max(),
                'mediana': valores.median()
            }
    
    # Criar DataFrame com resultados
    df_resultados = pd.DataFrame(estatisticas).T
    
    print("\n" + "="*60)
    print("RESULTADOS DA ANÁLISE")
    print("="*60)
    print("\nEstatísticas por métrica:")
    print(df_resultados.round(4))
    
    # Criar resumo adicional
    print("\n" + "="*60)
    print("RESUMO ADICIONAL")
    print("="*60)
    print(f"Número de runs analisadas: {len(dfs)}")
    print(f"Runs incluídas: {runs_encontradas}")
    
    # Verificar early stopping
    if 'early_stopped' in df_combined.columns:
        early_stopped_count = df_combined['early_stopped'].sum()
        print(f"Runs com early stopping: {early_stopped_count}/{len(dfs)}")
    
    # Melhor e pior run por best_val_dice
    if 'best_val_dice' in df_combined.columns and 'run_number' in df_combined.columns:
        melhor_idx = df_combined['best_val_dice'].idxmax()
        pior_idx = df_combined['best_val_dice'].idxmin()
        
        print(f"\nMelhor run (best_val_dice): Run {int(df_combined.loc[melhor_idx, 'run_number'])} - {df_combined.loc[melhor_idx, 'best_val_dice']:.4f}")
        print(f"Pior run (best_val_dice): Run {int(df_combined.loc[pior_idx, 'run_number'])} - {df_combined.loc[pior_idx, 'best_val_dice']:.4f}")
    
    # Salvar resultados em CSV
    output_file = base_path / "analise_runs_consolidada.csv"
    df_resultados.to_csv(output_file)
    print(f"\nResultados salvos em: {output_file}")
    
    # Salvar dados combinados
    combined_file = base_path / "dados_combinados_runs.csv"
    df_combined.to_csv(combined_file, index=False)
    print(f"Dados combinados salvos em: {combined_file}")
    
    return df_resultados

# ============================================
# EXECUÇÃO DIRETA - SEM LINHA DE COMANDO
# ============================================

# Defina o caminho do diretório aqui
diretorio ="/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/results/ViTUNet/ViTUNet_20260712_230901"

# Executar a análise
print("INICIANDO ANÁLISE DOS RUNS...")
print(f"Diretório: {diretorio}")
print("-" * 60)

resultados = analisar_runs(diretorio)

if resultados is not None:
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*60)
    
    # Exibir um resumo mais detalhado das médias
    print("\nMÉDIAS DAS PRINCIPAIS MÉTRICAS:")
    print("-" * 40)
    metricas_principais = ['best_val_dice', 'final_val_dice', 'final_val_accuracy', 'final_val_iou']
    for metrica in metricas_principais:
        if metrica in resultados.index:
            media = resultados.loc[metrica, 'media']
            dp = resultados.loc[metrica, 'desvio_padrao']
            print(f"{metrica:20s}: {media:.4f} ± {dp:.4f}")
else:
    print("\nFALHA NA ANÁLISE - Verifique o diretório")