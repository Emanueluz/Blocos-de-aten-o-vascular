#!/bin/bash

# ============================================
# SCRIPT PARA EXECUTAR TODOS OS MODELOS (GRAYSCALE)
# Segmentação de Vasos em Fundoscopia
# ============================================

# ============================================
# CONFIGURAÇÕES DOS DIRETÓRIOS
# ============================================

# --- DIRETÓRIOS DE TREINO (GRAYSCALE) ---
TRAIN_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/cinza"
TRAIN_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth"

# --- DIRETÓRIOS DE TESTE (GRAYSCALE) ---
TEST_BASES=(
    "/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/test/cinza"
    "/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg"
    "/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/test"
)

# --- DIRETÓRIO DE RESULTADOS ---
BASE_RESULTS_DIR="./results_all_models_grayscale"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ============================================
# PARÂMETROS DO MODELO
# ============================================
EPOCHS=50
RUNS=5
IMG_SIZE=224
PATIENCE=10
MIN_DELTA=0.001
NUM_WORKERS=4

# Batch sizes e learning rates por modelo (GRAYSCALE)
BATCH_EFFICIENTNET=16
LR_EFFICIENTNET=0.001

BATCH_RESNET=16
LR_RESNET=0.001

BATCH_SWIN=4
LR_SWIN=0.0001

BATCH_VGG=16
LR_VGG=0.001

BATCH_VIT=8
LR_VIT=0.001

BATCH_MOBILENET=16
LR_MOBILENET=0.001

# ============================================
# CONFIGURAÇÃO DO PYTHON
# ============================================

# Detectar comando Python disponível
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ERRO: Python não encontrado!"
    exit 1
fi

echo "✅ Usando Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version))"

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

# Função para imprimir cabeçalho
print_header() {
    local model=$1
    local total=$2
    local current=$3
    echo ""
    echo "######################################################################"
    echo "#                                                                     "
    echo "#  🚀 EXECUTANDO MODELO ${current}/${total}: ${model} (GRAYSCALE)"
    echo "#                                                                     "
    echo "######################################################################"
    echo ""
}

# Função para verificar diretórios
check_directories() {
    echo ""
    echo "🔍 VERIFICANDO DIRETÓRIOS..."
    echo "========================================="
    
    local has_error=0
    
    # Verificar diretórios de treino
    echo ""
    echo "📂 Diretórios de TREINO (GRAYSCALE):"
    if [ -d "${TRAIN_IMAGES_DIR}" ]; then
        local count=$(ls -1 "${TRAIN_IMAGES_DIR}" 2>/dev/null | wc -l)
        echo "  ✅ Imagens: ${TRAIN_IMAGES_DIR}"
        echo "     ${count} arquivos encontrados"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma imagem encontrada!"
            has_error=1
        fi
    else
        echo "  ❌ ERRO: Imagens não encontrado: ${TRAIN_IMAGES_DIR}"
        has_error=1
    fi
    
    if [ -d "${TRAIN_MASKS_DIR}" ]; then
        local count=$(ls -1 "${TRAIN_MASKS_DIR}" 2>/dev/null | wc -l)
        echo "  ✅ Máscaras: ${TRAIN_MASKS_DIR}"
        echo "     ${count} arquivos encontrados"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma máscara encontrada!"
            has_error=1
        fi
    else
        echo "  ❌ ERRO: Máscaras não encontrado: ${TRAIN_MASKS_DIR}"
        has_error=1
    fi
    
    # Verificar diretórios de teste
    echo ""
    echo "📂 Diretórios de TESTE (GRAYSCALE):"
    for test_dir in "${TEST_BASES[@]}"; do
        echo ""
        echo "  📁 $(basename "${test_dir}"):"
        echo "     Path: ${test_dir}"
        
        if [ ! -d "${test_dir}" ]; then
            echo "     ❌ Diretório não encontrado!"
            has_error=1
            continue
        fi
        
        # Verificar estruturas comuns
        if [ -d "${test_dir}/Original" ] && [ -d "${test_dir}/Ground truth" ]; then
            echo "     ✅ Estrutura: Original + Ground truth"
            echo "        Imagens: $(ls -1 "${test_dir}/Original" 2>/dev/null | wc -l) arquivos"
            echo "        Máscaras: $(ls -1 "${test_dir}/Ground truth" 2>/dev/null | wc -l) arquivos"
        elif [ -d "${test_dir}/images" ] && [ -d "${test_dir}/masks" ]; then
            echo "     ✅ Estrutura: images + masks"
            echo "        Imagens: $(ls -1 "${test_dir}/images" 2>/dev/null | wc -l) arquivos"
            echo "        Máscaras: $(ls -1 "${test_dir}/masks" 2>/dev/null | wc -l) arquivos"
        elif [ -d "${test_dir}/img" ] && [ -d "${test_dir}/mask" ]; then
            echo "     ✅ Estrutura: img + mask"
            echo "        Imagens: $(ls -1 "${test_dir}/img" 2>/dev/null | wc -l) arquivos"
            echo "        Máscaras: $(ls -1 "${test_dir}/mask" 2>/dev/null | wc -l) arquivos"
        elif [ -d "${test_dir}/cinza" ] && [ -d "${test_dir}/Ground truth" ]; then
            echo "     ✅ Estrutura: cinza + Ground truth"
            echo "        Imagens: $(ls -1 "${test_dir}/cinza" 2>/dev/null | wc -l) arquivos"
            echo "        Máscaras: $(ls -1 "${test_dir}/Ground truth" 2>/dev/null | wc -l) arquivos"
        else
            echo "     ⚠️  Estrutura não reconhecida"
            # Tentar encontrar arquivos
            local files=$(ls -1 "${test_dir}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | head -3)
            if [ -n "${files}" ]; then
                echo "        Total de arquivos de imagem: $(ls -1 "${test_dir}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)"
            fi
        fi
    done
    
    # Verificar diretório de resultados
    echo ""
    echo "📂 Diretório de RESULTADOS:"
    echo "  📁 ${BASE_RESULTS_DIR}"
    mkdir -p "${BASE_RESULTS_DIR}"
    echo "  ✅ Criado/verificado"
    
    echo ""
    echo "========================================="
    if [ ${has_error} -eq 0 ]; then
        echo "✅ Verificação concluída!"
    else
        echo "⚠️  Verificação concluída com AVISOS/ERROS!"
    fi
    echo ""
    
    return ${has_error}
}

# Função para executar um modelo
run_model() {
    local model_name=$1
    local script_path=$2
    local batch_size=$3
    local learning_rate=$4
    local current=$5
    local total=$6
    
    local results_dir="${BASE_RESULTS_DIR}/${model_name}_${TIMESTAMP}"
    local log_file="${results_dir}/execution.log"
    
    print_header "${model_name}" ${total} ${current}
    
    echo "📁 Resultados: ${results_dir}"
    echo "📝 Log: ${log_file}"
    echo "⏱️  Início: $(date)"
    echo ""
    
    # Criar diretório
    mkdir -p "${results_dir}"
    
    # Verificar se o script existe
    if [ ! -f "${script_path}" ]; then
        echo "❌ ERRO: Script não encontrado: ${script_path}"
        return 1
    fi
    
    # Construir lista de diretórios de teste
    local test_dirs=""
    for test_dir in "${TEST_BASES[@]}"; do
        test_dirs="${test_dirs} \"${test_dir}\""
    done
    
    # Construir comando completo
    local CMD="${PYTHON_CMD} \"${script_path}\" \
        --train_images_dir \"${TRAIN_IMAGES_DIR}\" \
        --train_masks_dir \"${TRAIN_MASKS_DIR}\" \
        --test_dirs ${test_dirs} \
        --input_mode grayscale \
        --epochs ${EPOCHS} \
        --n_runs ${RUNS} \
        --batch_size ${batch_size} \
        --img_size ${IMG_SIZE} \
        --learning_rate ${learning_rate} \
        --patience ${PATIENCE} \
        --min_delta ${MIN_DELTA} \
        --num_workers ${NUM_WORKERS} \
        --results_dir \"${results_dir}\" \
        --model_name \"${model_name}\""
    
    echo "📋 Comando:"
    echo "${CMD}"
    echo ""
    echo "========================================="
    
    # Executar e salvar log
    eval ${CMD} 2>&1 | tee "${log_file}"
    
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        echo ""
        echo "✅ ${model_name} concluído com SUCESSO!"
    else
        echo ""
        echo "❌ ${model_name} falhou com código ${exit_code}"
        echo "   Verifique o log: ${log_file}"
    fi
    
    echo "⏱️  Fim: $(date)"
    echo "========================================="
    
    return ${exit_code}
}

# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================

echo "######################################################################"
echo "#                                                                     "
echo "#  🚀 EXPERIMENTO COMPLETO - TODOS OS MODELOS (GRAYSCALE)             "
echo "#  Segmentação de Vasos em Fundoscopia                                "
echo "#                                                                     "
echo "######################################################################"
echo ""
echo "📊 Configurações Gerais:"
echo "  - Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1 | head -1))"
echo "  - Modo de entrada: GRAYSCALE (1 canal)"
echo "  - Épocas: ${EPOCHS}"
echo "  - Execuções por modelo: ${RUNS}"
echo "  - Tamanho da imagem: ${IMG_SIZE}x${IMG_SIZE}"
echo "  - Paciência: ${PATIENCE}"
echo "  - Diretório base: ${BASE_RESULTS_DIR}"
echo "  - Início: $(date)"
echo ""

# Verificar diretórios
check_directories

# Perguntar se deseja continuar
echo ""
read -p "🔄 Deseja continuar com a execução? (s/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "❌ Execução cancelada pelo usuário."
    exit 0
fi

# ============================================
# DEFINIÇÃO DOS MODELOS (6 MODELOS - GRAYSCALE)
# ============================================

# Formato: "Nome_do_Modelo" "caminho/do/script.py" batch_size learning_rate
MODELS=(
    "EfficientNetB0_UNet_Grayscale" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/efficientnet_grayscale_segmentation.py" ${BATCH_EFFICIENTNET} ${LR_EFFICIENTNET}
    "ResNet101_UNet_Grayscale" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/resnet101_grayscale_segmentation.py" ${BATCH_RESNET} ${LR_RESNET}
    "SwinUNet_Grayscale" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/swinunet_grayscale_segmentation.py" ${BATCH_SWIN} ${LR_SWIN}
    "VGG19_UNet_Grayscale" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/vgg19_grayscale_segmentation.py" ${BATCH_VGG} ${LR_VGG}
    "ViTUNet_Grayscale" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/vit_grayscale_segmentation.py" ${BATCH_VIT} ${LR_VIT}
    "MobileNetV2_UNet_Grayscale" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/mobilenetv2_grayscale_segmentation.py" ${BATCH_MOBILENET} ${LR_MOBILENET}
)

# Verificar se os scripts existem
echo ""
echo "🔍 Verificando scripts dos modelos (GRAYSCALE)..."
for ((i=0; i<${#MODELS[@]}; i+=4)); do
    SCRIPT_PATH=${MODELS[$((i+1))]}
    if [ -f "${SCRIPT_PATH}" ]; then
        echo "  ✅ $(basename "${SCRIPT_PATH}")"
    else
        echo "  ❌ $(basename "${SCRIPT_PATH}") - NÃO ENCONTRADO!"
        # Tentar encontrar arquivo similar
        DIR=$(dirname "${SCRIPT_PATH}")
        echo "     🔍 Arquivos similares em ${DIR}:"
        ls -la "${DIR}" 2>/dev/null | grep -i "grayscale" | awk '{print "        - " $9}'
    fi
done
echo ""

TOTAL_MODELOS=${#MODELS[@]}
CURRENT=0
SUCESSOS=0
FALHAS=0
MODELOS_FALHAS=()

# ============================================
# EXECUTAR MODELOS
# ============================================

for ((i=0; i<${#MODELS[@]}; i+=4)); do
    CURRENT=$((CURRENT + 1))
    MODEL_NAME=${MODELS[$i]}
    SCRIPT_PATH=${MODELS[$((i+1))]}
    BATCH_SIZE=${MODELS[$((i+2))]}
    LEARNING_RATE=${MODELS[$((i+3))]}
    
    # Verificar se o script existe antes de executar
    if [ ! -f "${SCRIPT_PATH}" ]; then
        echo ""
        echo "❌ PULANDO ${MODEL_NAME}: Script não encontrado: ${SCRIPT_PATH}"
        FALHAS=$((FALHAS + 1))
        MODELOS_FALHAS+=("${MODEL_NAME} (script não encontrado)")
        
        if [ ${CURRENT} -lt ${TOTAL_MODELOS} ]; then
            echo "⏳ Aguardando 5 segundos..."
            sleep 5
        fi
        continue
    fi
    
    run_model "${MODEL_NAME}" "${SCRIPT_PATH}" "${BATCH_SIZE}" "${LEARNING_RATE}" ${CURRENT} ${TOTAL_MODELOS}
    
    if [ $? -eq 0 ]; then
        SUCESSOS=$((SUCESSOS + 1))
    else
        FALHAS=$((FALHAS + 1))
        MODELOS_FALHAS+=("${MODEL_NAME}")
    fi
    
    # Pausa entre modelos
    if [ ${CURRENT} -lt ${TOTAL_MODELOS} ]; then
        echo ""
        echo "⏳ Aguardando 10 segundos antes do próximo modelo..."
        sleep 10
    fi
done

# ============================================
# RELATÓRIO FINAL
# ============================================

echo ""
echo "######################################################################"
echo "#                                                                     "
echo "#  📊 RELATÓRIO FINAL DO EXPERIMENTO (GRAYSCALE)                     "
echo "#                                                                     "
echo "######################################################################"
echo ""
echo "📈 Resumo:"
echo "  ✅ Modelos com sucesso: ${SUCESSOS}/${TOTAL_MODELOS}"
echo "  ❌ Modelos com falha: ${FALHAS}/${TOTAL_MODELOS}"
echo ""

if [ ${FALHAS} -gt 0 ]; then
    echo "⚠️  Modelos que falharam:"
    for model in "${MODELOS_FALHAS[@]}"; do
        echo "    - ${model}"
    done
    echo ""
    echo "📁 Para verificar os logs:"
    echo "   cd ${BASE_RESULTS_DIR}"
    echo "   grep -r 'ERROR' */execution.log"
fi

echo "📁 Resultados salvos em: ${BASE_RESULTS_DIR}/"
echo "🕐 Fim: $(date)"
echo ""

# Listar diretórios criados
echo "📂 Diretórios gerados:"
ls -la "${BASE_RESULTS_DIR}/" 2>/dev/null | grep "${TIMESTAMP}" | awk '{print "  - " $9}' || echo "  Nenhum diretório encontrado"

echo ""
echo "======================================================================"
if [ ${FALHAS} -eq 0 ]; then
    echo "🎉 TODOS OS MODELOS FORAM EXECUTADOS COM SUCESSO!"
    echo ""
    echo "📁 Para visualizar os resultados:"
    echo "   cd ${BASE_RESULTS_DIR}"
    echo "   ls -la"
    exit 0
else
    echo "⚠️  ALGUNS MODELOS FALHARAM. Verifique os logs para mais detalhes."
    exit 1
fi