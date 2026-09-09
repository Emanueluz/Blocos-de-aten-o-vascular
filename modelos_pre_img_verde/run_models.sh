#!/bin/bash

# ============================================
# SCRIPT PARA EXECUTAR TODOS OS MODELOS (verde)
# Segmentação de Vasos em Fundoscopia
# ============================================

# ============================================
# CONFIGURAÇÕES DOS DIRETÓRIOS
# ============================================

# --- DIRETÓRIOS DE TREINO (verde) ---
TRAIN_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/verde"
TRAIN_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth"

# --- DIRETÓRIOS DE TESTE (verde) ---
# TESTE 1: FIVES
TEST1_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/test/verde"
TEST1_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth"

# TESTE 2: Fundus-AVSeg
TEST2_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg/PDI_puro/verde"
TEST2_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg/Ground truth"

# TESTE 3: RETA
TEST3_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/PDI_puro/verde"
TEST3_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/Ground truth"

# --- DIRETÓRIO DE RESULTADOS ---
BASE_RESULTS_DIR="./results_verde"
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

# Batch sizes e learning rates por modelo (verde)
BATCH_EFFICIENTNET=16
LR_EFFICIENTNET=0.001

BATCH_RESNET=16
LR_RESNET=0.001

BATCH_SWIN=4
LR_SWIN=0.001

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
    echo "#  🚀 EXECUTANDO MODELO ${current}/${total}: ${model} (verde)"
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
    echo "📂 Diretórios de TREINO (verde):"
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
    echo "📂 Diretórios de TESTE (verde):"
    
    # Teste 1
    echo ""
    echo "  📁 TESTE 1 - FIVES:"
    echo "     Imagens: ${TEST1_IMAGES_DIR}"
    echo "     Máscaras: ${TEST1_MASKS_DIR}"
    
    if [ ! -d "${TEST1_IMAGES_DIR}" ]; then
        echo "     ❌ Diretório de imagens não encontrado!"
        has_error=1
    else
        local count=$(ls -1 "${TEST1_IMAGES_DIR}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
        echo "     ✅ Imagens: ${count} arquivos"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma imagem encontrada!"
            has_error=1
        fi
    fi
    
    if [ ! -d "${TEST1_MASKS_DIR}" ]; then
        echo "     ❌ Diretório de máscaras não encontrado!"
        has_error=1
    else
        local count=$(ls -1 "${TEST1_MASKS_DIR}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
        echo "     ✅ Máscaras: ${count} arquivos"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma máscara encontrada!"
            has_error=1
        fi
    fi
    
    # Teste 2
    echo ""
    echo "  📁 TESTE 2 - Fundus-AVSeg:"
    echo "     Imagens: ${TEST2_IMAGES_DIR}"
    echo "     Máscaras: ${TEST2_MASKS_DIR}"
    
    if [ ! -d "${TEST2_IMAGES_DIR}" ]; then
        echo "     ❌ Diretório de imagens não encontrado!"
        has_error=1
    else
        local count=$(ls -1 "${TEST2_IMAGES_DIR}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
        echo "     ✅ Imagens: ${count} arquivos"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma imagem encontrada!"
            has_error=1
        fi
    fi
    
    if [ ! -d "${TEST2_MASKS_DIR}" ]; then
        echo "     ❌ Diretório de máscaras não encontrado!"
        has_error=1
    else
        local count=$(ls -1 "${TEST2_MASKS_DIR}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
        echo "     ✅ Máscaras: ${count} arquivos"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma máscara encontrada!"
            has_error=1
        fi
    fi
    
    # Teste 3
    echo ""
    echo "  📁 TESTE 3 - RETA:"
    echo "     Imagens: ${TEST3_IMAGES_DIR}"
    echo "     Máscaras: ${TEST3_MASKS_DIR}"
    
    if [ ! -d "${TEST3_IMAGES_DIR}" ]; then
        echo "     ❌ Diretório de imagens não encontrado!"
        has_error=1
    else
        local count=$(ls -1 "${TEST3_IMAGES_DIR}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
        echo "     ✅ Imagens: ${count} arquivos"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma imagem encontrada!"
            has_error=1
        fi
    fi
    
    if [ ! -d "${TEST3_MASKS_DIR}" ]; then
        echo "     ❌ Diretório de máscaras não encontrado!"
        has_error=1
    else
        local count=$(ls -1 "${TEST3_MASKS_DIR}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
        echo "     ✅ Máscaras: ${count} arquivos"
        if [ ${count} -eq 0 ]; then
            echo "     ⚠️  ATENÇÃO: Nenhuma máscara encontrada!"
            has_error=1
        fi
    fi
    
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
    
    # Construir lista de diretórios de teste no formato imagem:máscara
    # para passar para --test_dirs
    TEST_DIRS_ARGS=""
    
    # Teste 1: FIVES
    TEST_DIRS_ARGS="${TEST_DIRS_ARGS} \"${TEST1_IMAGES_DIR}:${TEST1_MASKS_DIR}\""
    
    # Teste 2: Fundus-AVSeg
    TEST_DIRS_ARGS="${TEST_DIRS_ARGS} \"${TEST2_IMAGES_DIR}:${TEST2_MASKS_DIR}\""
    
    # Teste 3: RETA
    TEST_DIRS_ARGS="${TEST_DIRS_ARGS} \"${TEST3_IMAGES_DIR}:${TEST3_MASKS_DIR}\""
    
    # Construir comando completo usando --test_dirs
    local CMD="${PYTHON_CMD} \"${script_path}\" \
        --train_images_dir \"${TRAIN_IMAGES_DIR}\" \
        --train_masks_dir \"${TRAIN_MASKS_DIR}\" \
        --test_dirs ${TEST_DIRS_ARGS} \
        --input_mode verde \
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
echo "#  🚀 EXPERIMENTO COMPLETO - TODOS OS MODELOS (verde)             "
echo "#  Segmentação de Vasos em Fundoscopia                                "
echo "#                                                                     "
echo "######################################################################"
echo ""
echo "📊 Configurações Gerais:"
echo "  - Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1 | head -1))"
echo "  - Modo de entrada: verde (1 canal)"
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
# DEFINIÇÃO DOS MODELOS (6 MODELOS - verde)
# ============================================

# Formato: "Nome_do_Modelo" "caminho/do/script.py" batch_size learning_rate
MODELS=(
    "EfficientNetB0_UNet_verde" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/five_efficientNet_canal_verde.py" ${BATCH_EFFICIENTNET} ${LR_EFFICIENTNET}
    "ResNet101_UNet_verde" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/five_resnet101_canal_verde.py" ${BATCH_RESNET} ${LR_RESNET}
    "SwinUNet_verde" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/five_swin_canal_verde.py" ${BATCH_SWIN} ${LR_SWIN}
    "VGG19_UNet_verde" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/five_vgg19_canal_verde.py" ${BATCH_VGG} ${LR_VGG}
    "ViTUNet_verde" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/five_vit_canal_verde.py" ${BATCH_VIT} ${LR_VIT}
    "MobileNetV2_UNet_verde" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_verde/five_mobileNetV2net_canal_verde.py" ${BATCH_MOBILENET} ${LR_MOBILENET}
)

# Verificar se os scripts existem
echo ""
echo "🔍 Verificando scripts dos modelos (verde)..."
for ((i=0; i<${#MODELS[@]}; i+=4)); do
    SCRIPT_PATH=${MODELS[$((i+1))]}
    if [ -f "${SCRIPT_PATH}" ]; then
        echo "  ✅ $(basename "${SCRIPT_PATH}")"
    else
        echo "  ❌ $(basename "${SCRIPT_PATH}") - NÃO ENCONTRADO!"
        # Tentar encontrar arquivo similar
        DIR=$(dirname "${SCRIPT_PATH}")
        echo "     🔍 Arquivos similares em ${DIR}:"
        ls -la "${DIR}" 2>/dev/null | grep -i "verde" | awk '{print "        - " $9}'
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
echo "#  📊 RELATÓRIO FINAL DO EXPERIMENTO (verde)                     "
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