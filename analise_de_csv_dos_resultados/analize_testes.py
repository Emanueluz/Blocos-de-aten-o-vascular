import pandas as pd
import os

# Defina os caminhos e nomes dos modelos
modelos = [
    ("EfficientNetB0_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced/EfficientNetB0_UNet_g_enhanced/EfficientNetB0_UNet_g_enhanced_24-07-2026_17:01:57/test_results/aggregated_metrics_summary.csv"),
    ("MobileNetV2_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced/MobileNetV2_UNet_g_enhanced/MobileNetV2_UNet_g_enhanced_24_07_24_17:27:49/test_results/aggregated_metrics_summary.csv"),
    ("ResNet101_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced/ResNet101_UNet_g_enhanced/ResNet101_UNet_g_enhanced_24_07_24_17:51:05/test_results/aggregated_metrics_summary.csv"),
    ("SwinUNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced/SwinUNet_g_enhanced/SwinUNet_g_enhanced_25_07_25_11:06:56/test_results/aggregated_metrics_summary.csv"),
    ("VGG19_UNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced/VGG19_UNet_g_enhanced/VGG19_UNet_g_enhanced_24_07_24_18:09:05/test_results/aggregated_metrics_summary.csv"),
    ("ViTUNet", "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_g_enhanced/results-g_enhanced/ViTUNet_g_enhanced/ViTUNet_g_enhanced_24_07_24_18:16:02/test_results/aggregated_metrics_summary.csv")
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
df_consolidado.to_csv('/home/emanuel/Documentos/mestrado/treino dos modelos/analise_de_csv_dos_resultados/dados_de_teste/G_CINZA_consolidated_metrics.csv', index=False)
print("Arquivo 'G_consolidated_metrics.csv' gerado com sucesso!")