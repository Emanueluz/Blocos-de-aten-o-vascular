import os
import pandas as pd
import numpy as np
from pathlib import Path

def analisar_runs(diretorio_base, pasta_saida_base, nome_modelo):
    """
    Analisa os arquivos run_summary.csv nos subdiretórios run_0 a run_4

    Args:
        diretorio_base (str ou Path): Caminho para o diretório que contém as pastas run_*
        pasta_saida_base (str ou Path): Caminho base onde serão salvos os resultados
        nome_modelo (str): Nome do modelo (será usado para criar subpastas)

    Returns:
        DataFrame com estatísticas consolidadas ou None se não encontrar dados.
    """
    # Converter para Path para facilitar manipulação
    diretorio_base = Path(diretorio_base)
    pasta_saida_base = Path(pasta_saida_base)

    # Verificar se o diretório de entrada existe
    if not diretorio_base.exists():
        print(f"Erro: Diretório {diretorio_base} não encontrado!")
        return None

    # Lista para armazenar os DataFrames
    dfs = []
    runs_encontradas = []

    # Procurar diretórios run_0 a run_4
    for i in range(5):
        run_dir = diretorio_base / f"run_{i}"
        csv_file = run_dir / "run_summary.csv"

        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
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

    # Selecionar colunas para análise (apenas as que existem)
    colunas_interesse = [
        'best_val_dice', 'best_epoch', 'total_epochs',
        'total_training_time', 'final_val_dice', 'final_val_loss',
        'final_val_accuracy', 'final_val_iou',
        'final_val_sensitivity', 'final_val_specificity'
    ]
    colunas_existentes = [col for col in colunas_interesse if col in df_combined.columns]

    # Calcular estatísticas
    estatisticas = {}
    for coluna in colunas_existentes:
        df_combined[coluna] = pd.to_numeric(df_combined[coluna], errors='coerce')
        valores = df_combined[coluna].dropna()
        if len(valores) > 0:
            estatisticas[coluna] = {
                'media': valores.mean(),
                'desvio_padrao': valores.std(),
                'min': valores.min(),
                'max': valores.max(),
                'mediana': valores.median()
            }

    df_resultados = pd.DataFrame(estatisticas).T

    print("\n" + "="*60)
    print("RESULTADOS DA ANÁLISE")
    print("="*60)
    print("\nEstatísticas por métrica:")
    print(df_resultados.round(4))

    print("\n" + "="*60)
    print("RESUMO ADICIONAL")
    print("="*60)
    print(f"Número de runs analisadas: {len(dfs)}")
    print(f"Runs incluídas: {runs_encontradas}")

    if 'early_stopped' in df_combined.columns:
        early_stopped_count = df_combined['early_stopped'].sum()
        print(f"Runs com early stopping: {early_stopped_count}/{len(dfs)}")

    if 'best_val_dice' in df_combined.columns and 'run_number' in df_combined.columns:
        melhor_idx = df_combined['best_val_dice'].idxmax()
        pior_idx = df_combined['best_val_dice'].idxmin()
        print(f"\nMelhor run (best_val_dice): Run {int(df_combined.loc[melhor_idx, 'run_number'])} - {df_combined.loc[melhor_idx, 'best_val_dice']:.4f}")
        print(f"Pior run (best_val_dice): Run {int(df_combined.loc[pior_idx, 'run_number'])} - {df_combined.loc[pior_idx, 'best_val_dice']:.4f}")

    # Criar subdiretório para o modelo (se não existir)
    pasta_modelo = pasta_saida_base / nome_modelo
    pasta_modelo.mkdir(parents=True, exist_ok=True)

    # Salvar resultados
    output_file = pasta_modelo / "analise_runs_consolidada.csv"
    df_resultados.to_csv(output_file)
    print(f"\nResultados salvos em: {output_file}")

    combined_file = pasta_modelo / "dados_combinados_runs.csv"
    df_combined.to_csv(combined_file, index=False)
    print(f"Dados combinados salvos em: {combined_file}")

    return df_resultados


# ============================================
# EXECUÇÃO DIRETA
# ============================================

# Defina os caminhos
base_entrada = "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza"
base_saida = "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_combinados_cinza"

# Dicionário com os modelos e seus diretórios
modelos = {
    "EfficientNetB0_UNet_cinza": f"{base_entrada}/EfficientNetB0_UNet_cinza/EfficientNetB0_UNet_cinza_14-07-2026_13:36:22",
    "MobileNetV2_UNet_cinza": f"{base_entrada}/MobileNetV2_UNet_cinza/MobileNetV2_UNet_cinza_14_07_14_13:58:51",
    "ResNet101_UNet_cinza": f"{base_entrada}/ResNet101_UNet_cinza/ResNet101_UNet_cinza_14_07_14_14:24:11",
    "SwinUNet_cinza": f"{base_entrada}/SwinUNet_cinza/SwinUNet_cinza_14_07_14_22:10:42",
    "VGG19_UNet_cinza": f"{base_entrada}/VGG19_UNet_Grayscale/VGG19_UNet_Grayscale_14_07_14_14:50:49",
    "ViTUNet_cinza": f"{base_entrada}/ViTUNet_cinza/ViTUNet_cinza_14_07_14_17:53:48",
}

print("INICIANDO ANÁLISE DOS RUNS...")
print("-" * 60)

# Armazenar todos os resultados para consolidação final (opcional)
todos_resultados = {}

for nome, caminho in modelos.items():
    print(f"\n--- Analisando {nome} ---")
    resultado = analisar_runs(caminho, base_saida, nome)
    if resultado is not None:
        todos_resultados[nome] = resultado

# Opcional: consolidar todos os modelos em um único DataFrame
if todos_resultados:
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*60)
    # Exemplo: média do best_val_dice por modelo
    for nome, df in todos_resultados.items():
        if 'best_val_dice' in df.index:
            media = df.loc['best_val_dice', 'media']
            dp = df.loc['best_val_dice', 'desvio_padrao']
            print(f"{nome:30s} best_val_dice = {media:.4f} ± {dp:.4f}")
else:
    print("\nFALHA NA ANÁLISE - Verifique os diretórios")