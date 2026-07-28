import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Métricas de interesse (basta o nome base)
METRICAS = [
    'dice',
    'accuracy',
    'iou',
    'sensitivity',
    'specificity'
]

# Diretório onde serão salvos os resumos consolidados
DIRETORIO_SAIDA = "/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/csv_resumo_dos_testes"

# ============================================================
# FUNÇÃO PARA EXTRAIR TIMESTAMP DE NOME DE PASTA (MÚLTIPLOS FORMATOS)
# ============================================================
def extrair_timestamp_pasta(nome_pasta):
    """
    Tenta extrair timestamp do final do nome da pasta.
    Suporta os formatos observados:
      - _DD-MM-YYYY_HH:MM:SS
      - _DD_MM_YY_HH:MM:SS  (ano com 2 dígitos)
      - _AAAAMMDD_HHMMSS
      - _AAAAMMDD_HHMMSS (sem separadores)
    Retorna um objeto datetime ou None.
    """
    # Remove o nome base (ex: "ResNet101_UNet_g_enhanced") para ficar apenas com o timestamp
    # Vamos extrair a parte após o último underscore que pareça uma data/hora
    padroes = [
        # Formato 1: _DD-MM-YYYY_HH:MM:SS
        (r'_(\d{2}-\d{2}-\d{4}_\d{2}:\d{2}:\d{2})$', '%d-%m-%Y_%H:%M:%S'),
        # Formato 2: _DD_MM_YY_HH:MM:SS (ano com 2 dígitos)
        (r'_(\d{2}_\d{2}_\d{2}_\d{2}:\d{2}:\d{2})$', '%d_%m_%y_%H:%M:%S'),
        # Formato 3: _AAAAMMDD_HHMMSS (sem separadores)
        (r'_(\d{8}_\d{6})$', '%Y%m%d_%H%M%S'),
        # Formato 4: _AAAAMMDD_HHMMSS (com hífen? mas difícil)
    ]
    for padrao, fmt in padroes:
        match = re.search(padrao, nome_pasta)
        if match:
            timestamp_str = match.group(1)
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
    return None

# ============================================================
# FUNÇÃO PARA OBTER A PASTA DE EXECUÇÃO MAIS RECENTE PARA UM MODELO
# ============================================================
def obter_pasta_mais_recente(modelo_dir):
    """
    Dado o diretório de um modelo, procura subpastas com timestamp
    (usando múltiplos formatos). Retorna o caminho da pasta mais recente
    ou None se nenhuma for encontrada.
    Como fallback, usa a data de modificação do sistema.
    """
    pastas_com_ts = []
    for item in modelo_dir.iterdir():
        if item.is_dir():
            ts = extrair_timestamp_pasta(item.name)
            if ts is not None:
                pastas_com_ts.append((ts, item))
            else:
                # Se não conseguir extrair, tenta usar a data de modificação
                # (fallback) - mas vamos priorizar as que têm timestamp no nome
                pass

    if not pastas_com_ts:
        # Fallback: pega a pasta mais recentemente modificada
        print(f"   ⚠️  Nenhum timestamp reconhecido em {modelo_dir.name}. Usando data de modificação.")
        pastas = [p for p in modelo_dir.iterdir() if p.is_dir()]
        if not pastas:
            return None
        # Retorna a pasta com a data de modificação mais recente
        return max(pastas, key=lambda p: p.stat().st_mtime)

    # Ordena por timestamp (mais recente primeiro)
    pastas_com_ts.sort(key=lambda x: x[0], reverse=True)
    return pastas_com_ts[0][1]  # caminho da pasta mais recente

# ============================================================
# FUNÇÃO PARA RESUMIR RESULTADOS DE TESTE DE UM DIRETÓRIO
# ============================================================
def resumir_resultados_testes(diretorio_base):
    """
    Recebe o diretório base (ex: .../results_cinza).
    Para cada subpasta de modelo, encontra a execução mais recente,
    lê aggregated_metrics_summary.csv e calcula:
      - média da média (across runs)
      - desvio padrão da média (variação entre runs)
      para cada métrica listada em METRICAS.
    Retorna um DataFrame com uma linha por modelo.
    """
    base_path = Path(diretorio_base)
    if not base_path.exists():
        raise FileNotFoundError(f"Diretório base não encontrado: {diretorio_base}")

    resultados = {}
    # Itera sobre cada subdiretório (cada modelo)
    for modelo_dir in base_path.iterdir():
        if not modelo_dir.is_dir():
            continue

        # Obtém a pasta de execução mais recente
        pasta_exec = obter_pasta_mais_recente(modelo_dir)
        if pasta_exec is None:
            print(f"⚠️  Nenhuma pasta de execução encontrada para: {modelo_dir.name}")
            continue

        csv_path = pasta_exec / "test_results" / "aggregated_metrics_summary.csv"
        if not csv_path.exists():
            print(f"⚠️  CSV não encontrado: {csv_path}")
            continue

        # Lê o CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Erro ao ler {csv_path}: {e}")
            continue

        if df.empty:
            print(f"⚠️  CSV vazio: {csv_path}")
            continue

        # Para cada métrica, extrai a média e o desvio da coluna _mean
        registro = {}
        for metrica in METRICAS:
            col_mean = f"{metrica}_mean"
            if col_mean in df.columns:
                media_media = df[col_mean].mean()
                std_media = df[col_mean].std()
                registro[f"{metrica}_mean_mean"] = media_media
                registro[f"{metrica}_mean_std"] = std_media
            else:
                print(f"⚠️  Coluna '{col_mean}' não encontrada em {csv_path}")
                # Não interrompe, apenas não adiciona essa métrica

        if not registro:  # Se não encontrou nenhuma métrica
            print(f"⚠️  Nenhuma métrica reconhecida em {csv_path}")
            continue

        registro['num_runs'] = len(df)
        registro['modelo'] = modelo_dir.name
        resultados[modelo_dir.name] = registro

    # Converte para DataFrame
    if not resultados:
        return pd.DataFrame()  # vazio

    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    # Reordena colunas
    colunas_metricas = []
    for metrica in METRICAS:
        colunas_metricas.append(f"{metrica}_mean_mean")
        colunas_metricas.append(f"{metrica}_mean_std")
    colunas_ordenadas = ['modelo'] + colunas_metricas + ['num_runs']
    colunas_existentes = [col for col in colunas_ordenadas if col in df_resultados.columns]
    df_resultados = df_resultados[colunas_existentes]

    return df_resultados

# ============================================================
# FUNÇÃO PARA SALVAR O RESUMO EM CSV
# ============================================================
def salvar_resumo(df, nome_arquivo, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    caminho_completo = output_path / nome_arquivo
    df.to_csv(caminho_completo, index=False)
    print(f"✅ Resumo salvo em: {caminho_completo}")

# ============================================================
# EXEMPLO DE USO NA MAIN
# ============================================================
if __name__ == "__main__":
    # Lista de diretórios base (conforme solicitado)
    diretorios_para_processar = [
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/results",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/results-verde",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced",
        "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde_clahe/results-verde_clahe",  # duplicado, mas ok
    ]

    Path(DIRETORIO_SAIDA).mkdir(parents=True, exist_ok=True)

    dfs = []
    for dir_path in diretorios_para_processar:
        print(f"\n📁 Processando diretório: {dir_path}")
        try:
            df_temp = resumir_resultados_testes(dir_path)
            if df_temp.empty:
                print(f"⚠️  Nenhum dado encontrado em {dir_path}")
                continue
            # Adiciona coluna de origem
            nome_base = Path(dir_path).name
            df_temp['diretorio_origem'] = nome_base
            dfs.append(df_temp)
        except Exception as e:
            print(f"❌ Erro ao processar {dir_path}: {e}")

    if not dfs:
        print("⚠️  Nenhum dado foi processado. Verifique os diretórios e os CSVs.")
    else:
        df_consolidado = pd.concat(dfs, ignore_index=True)
        df_consolidado = df_consolidado.sort_values('modelo').reset_index(drop=True)

        print("\n" + "="*80)
        print("📊 RESUMO CONSOLIDADO DOS TESTES (MÉDIA E DESVIO ENTRE RUNS)")
        print("="*80)
        print(df_consolidado.round(4).to_string())

        # Salva consolidado
        salvar_resumo(df_consolidado, "resumo_testes_consolidado.csv", DIRETORIO_SAIDA)

        # Salva separado por diretório de origem
        for nome_dir, grupo in df_consolidado.groupby('diretorio_origem'):
            nome_arquivo = f"resumo_testes_{nome_dir}.csv"
            salvar_resumo(grupo.drop(columns=['diretorio_origem']), nome_arquivo, DIRETORIO_SAIDA)

        print("\n🎯 Processamento completo!")