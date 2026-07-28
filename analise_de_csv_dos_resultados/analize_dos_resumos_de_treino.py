import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Colunas que deseja extrair (médias e desvios)
COLUNAS_METRICAS = [
    'best_val_dice',
    'final_val_accuracy',
    'final_val_iou',
    'final_val_sensitivity',
    'final_val_specificity',
    'total_training_time'
]

# Diretório onde serão salvos os resumos consolidados
DIRETORIO_SAIDA = "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/csv_resumo_dos_treinos"

# ============================================================
# FUNÇÃO PARA EXTRAIR TIMESTAMP DO NOME DO ARQUIVO
# ============================================================
def extrair_timestamp(nome_arquivo):
    padrao = r'_(\d{8}_\d{6})\.csv$'
    match = re.search(padrao, nome_arquivo)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None

# ============================================================
# FUNÇÃO PARA RESUMIR UM ÚNICO DIRETÓRIO DE MODELOS
# ============================================================
def resumir_resultados(diretorio_base):
    """
    Recebe o caminho de um diretório que contém subpastas de modelos.
    Cada subpasta deve conter uma pasta 'RELATORIO_DAS_EXECUCOES'
    com arquivos 'RESUMO_EXECUCOES_*.csv'.
    Para cada modelo, seleciona o CSV mais recente (pelo timestamp no nome)
    e calcula a média e o desvio padrão das colunas especificadas em COLUNAS_METRICAS.
    Retorna um DataFrame com uma linha por modelo, contendo médias e desvios.
    """
    resultados = {}
    base_path = Path(diretorio_base)

    if not base_path.exists():
        raise FileNotFoundError(f"Diretório base não encontrado: {diretorio_base}")

    # Itera sobre cada subdiretório (cada modelo)
    for modelo_dir in base_path.iterdir():
        if not modelo_dir.is_dir():
            continue

        relatorio_dir = modelo_dir / "RELATORIO_DAS_EXECUCOES"
        if not relatorio_dir.exists() or not relatorio_dir.is_dir():
            print(f"⚠️  Pasta RELATORIO_DAS_EXECUCOES não encontrada em: {modelo_dir.name}")
            continue

        arquivos_csv = list(relatorio_dir.glob("RESUMO_EXECUCOES_*.csv"))
        if not arquivos_csv:
            print(f"⚠️  Nenhum CSV de resumo encontrado em: {relatorio_dir}")
            continue

        # Escolhe o CSV mais recente
        arquivo_mais_recente = None
        timestamp_mais_recente = None
        for csv_path in arquivos_csv:
            ts = extrair_timestamp(csv_path.name)
            if ts is not None:
                if timestamp_mais_recente is None or ts > timestamp_mais_recente:
                    timestamp_mais_recente = ts
                    arquivo_mais_recente = csv_path

        if arquivo_mais_recente is None:
            print(f"⚠️  Não foi possível extrair timestamp de nenhum CSV em: {relatorio_dir}")
            continue

        # Lê o CSV
        try:
            df = pd.read_csv(arquivo_mais_recente)
        except Exception as e:
            print(f"❌ Erro ao ler {arquivo_mais_recente}: {e}")
            continue

        # Verifica quais colunas existem
        colunas_presentes = [col for col in COLUNAS_METRICAS if col in df.columns]
        if not colunas_presentes:
            print(f"⚠️  Nenhuma das colunas esperadas encontradas em {arquivo_mais_recente.name}")
            continue

        # Calcula a média e o desvio padrão para cada métrica
        registro = {}
        for col in colunas_presentes:
            registro[f"{col}_mean"] = df[col].mean()
            registro[f"{col}_std"] = df[col].std()

        # Adiciona o nome do modelo e a quantidade de execuções
        registro['modelo'] = modelo_dir.name
        registro['num_runs'] = len(df)

        resultados[modelo_dir.name] = registro

    # Converte dicionário para DataFrame
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')

    # Reordena colunas: modelo, num_runs, depois as métricas (mean e std alternados)
    colunas_ordenadas = ['modelo', 'num_runs']
    for col in COLUNAS_METRICAS:
        if f"{col}_mean" in df_resultados.columns:
            colunas_ordenadas.append(f"{col}_mean")
            colunas_ordenadas.append(f"{col}_std")

    # Mantém apenas as colunas que existem no DataFrame
    colunas_existentes = [c for c in colunas_ordenadas if c in df_resultados.columns]
    df_resultados = df_resultados[colunas_existentes]

    return df_resultados

# ============================================================
# FUNÇÃO PARA SALVAR O RESUMO EM CSV
# ============================================================
def salvar_resumo(df, nome_arquivo, output_dir):
    """
    Salva um DataFrame em um arquivo CSV no diretório de saída.
    Cria o diretório se não existir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    caminho_completo = output_path / nome_arquivo
    df.to_csv(caminho_completo, index=False)
    print(f"✅ Resumo salvo em: {caminho_completo}")

# ============================================================
# EXEMPLO DE USO NA MAIN
# ============================================================
if __name__ == "__main__":
    # Lista de diretórios que você deseja processar
    diretorios_para_processar = [
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/results",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/results-verde",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde_clahe/results-verde_clahe",
    ]

    # Cria o diretório de saída se não existir
    Path(DIRETORIO_SAIDA).mkdir(parents=True, exist_ok=True)

    # Processa cada diretório e armazena os DataFrames
    dfs = []
    for dir_path in diretorios_para_processar:
        print(f"\n📁 Processando diretório: {dir_path}")
        try:
            df_temp = resumir_resultados(dir_path)
            if df_temp.empty:
                print(f"⚠️  Nenhum dado encontrado em {dir_path}")
                continue
            # Adiciona uma coluna indicando a origem (nome do diretório base)
            nome_base = Path(dir_path).name
            df_temp['diretorio_origem'] = nome_base
            dfs.append(df_temp)
        except Exception as e:
            print(f"❌ Erro ao processar {dir_path}: {e}")

    if not dfs:
        print("⚠️  Nenhum dado foi processado. Verifique os diretórios.")
    else:
        # Concatena todos os DataFrames em um único
        df_consolidado = pd.concat(dfs, ignore_index=True)

        # Ordena por modelo (opcional)
        df_consolidado = df_consolidado.sort_values('modelo').reset_index(drop=True)

        # Exibe no terminal (opcional)
        print("\n" + "="*80)
        print("📊 RESUMO CONSOLIDADO DE TODOS OS DIRETÓRIOS (MÉDIA ± DESVIO)")
        print("="*80)
        print(df_consolidado.round(4).to_string())

        # Salva o consolidado
        salvar_resumo(df_consolidado, "resumo_consolidado_todos_modelos.csv", DIRETORIO_SAIDA)

        # Salvar também separadamente por diretório de origem
        for nome_dir, grupo in df_consolidado.groupby('diretorio_origem'):
            nome_arquivo = f"resumo_{nome_dir}.csv"
            # Remove a coluna de origem para não poluir o arquivo separado
            grupo_sem_origem = grupo.drop(columns=['diretorio_origem'])
            salvar_resumo(grupo_sem_origem, nome_arquivo, DIRETORIO_SAIDA)

        print("\n🎯 Processamento completo!")