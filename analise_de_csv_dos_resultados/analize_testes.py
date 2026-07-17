import pandas as pd
import os

# Defina os caminhos e nomes dos modelos
modelos = [
    ("EfficientNetB0_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/EfficientNetB0_UNet_cinza/EfficientNetB0_UNet_cinza_14-07-2026_13:36:22/test_results/aggregated_metrics_summary.csv"),
    ("MobileNetV2_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/MobileNetV2_UNet_cinza/MobileNetV2_UNet_cinza_14_07_14_13:58:51/test_results/aggregated_metrics_summary.csv"),
    ("ResNet101_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/ResNet101_UNet_cinza/ResNet101_UNet_cinza_14_07_14_14:24:11/test_results/aggregated_metrics_summary.csv"),
    ("SwinUNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/SwinUNet_cinza/SwinUNet_cinza_14_07_14_22:10:42/test_results/aggregated_metrics_summary.csv"),
    ("VGG19_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/VGG19_UNet_Grayscale/VGG19_UNet_Grayscale_14_07_14_14:50:49/test_results/aggregated_metrics_summary.csv"),
    ("ViTUNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/results_cinza/ViTUNet_cinza/ViTUNet_cinza_14_07_14_17:53:48/test_results/aggregated_metrics_summary.csv")
]

# Lista para armazenar os dados agregados
dados_consolidados = []

for nome_modelo, caminho in modelos:
    # Lê o CSV
    df = pd.read_csv(caminho)
    
    # Seleciona apenas colunas numéricas (exclui 'run_id' se existir)
    colunas_numericas = df.select_dtypes(include='number').columns
    # Remove 'run_id' se presente
    if 'run_id' in colunas_numericas:
        colunas_numericas = colunas_numericas.drop('run_id')
    
    # Calcula média e desvio padrão de cada coluna (entre as 5 execuções)
    medias = df[colunas_numericas].mean()
    desvios = df[colunas_numericas].std()
    
    # Monta a linha do modelo com os resultados
    linha = {'model_name': nome_modelo}
    for col in medias.index:
        linha[f'{col}_media'] = medias[col]      # média das médias das execuções
        linha[f'{col}_desvio'] = desvios[col]    # desvio padrão entre execuções
    
    dados_consolidados.append(linha)

# Cria DataFrame final
df_consolidado = pd.DataFrame(dados_consolidados)

# Salva o CSV consolidado
df_consolidado.to_csv('/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_de_teste/CINZA_consolidated_metrics.csv', index=False)
print("Arquivo 'consolidated_metrics.csv' gerado com sucesso!")