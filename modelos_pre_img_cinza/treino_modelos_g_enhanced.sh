#!/bin/bash

# ============================================
# SCRIPT PARA EXECUTAR TODOS OS MODELOS (GRAYSCALE)
# Segmentação de Vasos em Fundoscopia
# ============================================
# Estrutura de saída por modelo:
#   <results_dir>/
#     ├── saved_models/
#     ├── run_0/ ... run_N/
#     ├── test_results/
#     │    ├── FIVES/                      <- relatórios independentes
#     │    │    ├── REPORT.txt
#     │    │    ├── aggregated_metrics_summary.csv
#     │    │    ├── aggregated_statistics_summary.csv
#     │    │    └── consolidated_per_image_metrics.csv
#     │    ├── Fundus-AVSeg/               <- relatórios independentes
#     │    │    └── ...
#     │    ├── RETA/                       <- relatórios independentes
#     │    │    └── ...
#     │    └── COMPARATIVO_ENTRE_BASES/    <- comparativo geral
#     │         ├── comparativo_entre_bases.csv
#     │         ├── COMPARATIVO_ENTRE_BASES.txt
#     │         └── comparativo_entre_bases.png
# ============================================

# ============================================
# CONFIGURAÇÕES DOS DIRETÓRIOS
# ============================================

# --- DIRETÓRIOS DE TREINO (GRAYSCALE) ---
TRAIN_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/train/g_enhanced"
TRAIN_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/train/Ground truth"

# --- DIRETÓRIOS DE TESTE (GRAYSCALE) ---
# Cada base de teste será passada separadamente para o script Python
# no formato: "<imagens>:<mascaras>"
# Assim o Python gera relatórios SEPARADOS por base + comparativo final.

# TESTE 1: FIVES
TEST1_NAME="FIVES"
TEST1_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/PDI_puro/test/g_enhanced"
TEST1_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/FIVES/test/Ground truth"

# TESTE 2: Fundus-AVSeg
TEST2_NAME="Fundus-AVSeg"
TEST2_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg/PDI_puro/g_enhanced"
TEST2_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/Fundus-AVSeg/Ground truth"

# TESTE 3: RETA
TEST3_NAME="RETA"
EST3_IMAGES_DIR="/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/PDI_puro/g_enhanced"
TEST3_MASKS_DIR="/home/emanuel/Documentos/mestrado/bases de dados/RETA/images/train/Ground truth"
# --- DIRETÓRIO DE RESULTADOS ---
BASE_RESULTS_DIR="./results_g_enhanced"

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
    
    # -------- TREINO --------
    echo ""
    echo "📂 Diretórios de TREINO (GRAYSCALE):"
    if [ -d "${TRAIN_IMAGES_DIR}" ]; then
        local count=$(ls -1 "${TRAIN_IMAGES_DIR}" 2>/dev/null | wc -l)
        echo "  ✅ Imagens: ${TRAIN_IMAGES_DIR}"
        echo "     ${count} arquivos encontrados"
        [ ${count} -eq 0 ] && { echo "     ⚠️  Nenhuma imagem encontrada!"; has_error=1; }
    else
        echo "  ❌ ERRO: Imagens não encontrado: ${TRAIN_IMAGES_DIR}"
        has_error=1
    fi
    
    if [ -d "${TRAIN_MASKS_DIR}" ]; then
        local count=$(ls -1 "${TRAIN_MASKS_DIR}" 2>/dev/null | wc -l)
        echo "  ✅ Máscaras: ${TRAIN_MASKS_DIR}"
        echo "     ${count} arquivos encontrados"
        [ ${count} -eq 0 ] && { echo "     ⚠️  Nenhuma máscara encontrada!"; has_error=1; }
    else
        echo "  ❌ ERRO: Máscaras não encontrado: ${TRAIN_MASKS_DIR}"
        has_error=1
    fi
    
    # -------- TESTES --------
    echo ""
    echo "📂 Diretórios de TESTE (GRAYSCALE) — cada base gera relatório SEPARADO:"
    
    # Teste 1
    echo ""
    echo "  📁 TESTE 1 - ${TEST1_NAME}:"
    echo "     Imagens:  ${TEST1_IMAGES_DIR}"
    echo "     Máscaras: ${TEST1_MASKS_DIR}"
    check_test_dir "${TEST1_IMAGES_DIR}" "${TEST1_MASKS_DIR}" "imagens" "máscaras"
    [ $? -ne 0 ] && has_error=1
    
    # Teste 2
    echo ""
    echo "  📁 TESTE 2 - ${TEST2_NAME}:"
    echo "     Imagens:  ${TEST2_IMAGES_DIR}"
    echo "     Máscaras: ${TEST2_MASKS_DIR}"
    check_test_dir "${TEST2_IMAGES_DIR}" "${TEST2_MASKS_DIR}" "imagens" "máscaras"
    [ $? -ne 0 ] && has_error=1
    
    # Teste 3
    echo ""
    echo "  📁 TESTE 3 - ${TEST3_NAME}:"
    echo "     Imagens:  ${TEST3_IMAGES_DIR}"
    echo "     Máscaras: ${TEST3_MASKS_DIR}"
    check_test_dir "${TEST3_IMAGES_DIR}" "${TEST3_MASKS_DIR}" "imagens" "máscaras"
    [ $? -ne 0 ] && has_error=1
    
    # -------- RESULTADOS --------
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

# Função auxiliar para checar um par imagens/máscaras
check_test_dir() {
    local img_dir=$1
    local mask_dir=$2
    local label_img=$3
    local label_mask=$4
    local local_err=0
    
    if [ ! -d "${img_dir}" ]; then
        echo "     ❌ Diretório de ${label_img} não encontrado!"
        return 1
    fi
    local count=$(ls -1 "${img_dir}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
    echo "     ✅ ${label_img}: ${count} arquivos"
    [ ${count} -eq 0 ] && { echo "     ⚠️  Nenhuma ${label_img} encontrada!"; local_err=1; }
    
    if [ ! -d "${mask_dir}" ]; then
        echo "     ❌ Diretório de ${label_mask} não encontrado!"
        return 1
    fi
    count=$(ls -1 "${mask_dir}" 2>/dev/null | grep -E "\.(png|jpg|jpeg|tif|tiff|bmp)$" | wc -l)
    echo "     ✅ ${label_mask}: ${count} arquivos"
    [ ${count} -eq 0 ] && { echo "     ⚠️  Nenhuma ${label_mask} encontrada!"; local_err=1; }
    
    return ${local_err}
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
    echo "📝 Log:        ${log_file}"
    echo "⏱️  Início:     $(date)"
    echo ""
    echo "🗂️  Bases de teste que serão processadas SEPARADAMENTE:"
    echo "   1) ${TEST1_NAME}"
    echo "   2) ${TEST2_NAME}"
    echo "   3) ${TEST3_NAME}"
    echo "   ➜ Relatórios individuais em: ${results_dir}/<EXPERIMENT>/test_results/<BASE>/"
    echo "   ➜ Comparativo geral em:      ${results_dir}/<EXPERIMENT>/test_results/COMPARATIVO_ENTRE_BASES/"
    echo ""
    
    mkdir -p "${results_dir}"
    
    if [ ! -f "${script_path}" ]; then
        echo "❌ ERRO: Script não encontrado: ${script_path}"
        return 1
    fi
    
    # ---------- Montagem dos --test_dirs ----------
    # Cada base vai como "<imagens>:<mascaras>"
    # O Python detecta o nome da base automaticamente (usa o basename do diretório de imagens).
    # Para garantir nomes estáveis (FIVES / Fundus-AVSeg / RETA), usamos os nomes definidos acima.
    TEST_DIRS_ARGS=""
    TEST_DIRS_ARGS="${TEST_DIRS_ARGS} \"${TEST1_IMAGES_DIR}:${TEST1_MASKS_DIR}\""
    TEST_DIRS_ARGS="${TEST_DIRS_ARGS} \"${TEST2_IMAGES_DIR}:${TEST2_MASKS_DIR}\""
    TEST_DIRS_ARGS="${TEST_DIRS_ARGS} \"${TEST3_IMAGES_DIR}:${TEST3_MASKS_DIR}\""
    
    # ---------- Comando Python ----------
    local CMD="${PYTHON_CMD} \"${script_path}\" \
        --train_images_dir \"${TRAIN_IMAGES_DIR}\" \
        --train_masks_dir  \"${TRAIN_MASKS_DIR}\" \
        --test_dirs ${TEST_DIRS_ARGS} \
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
    
    eval ${CMD} 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    
    if [ ${exit_code} -eq 0 ]; then
        echo ""
        echo "✅ ${model_name} concluído com SUCESSO!"
        echo ""
        echo "📊 Relatórios gerados para ${model_name}:"
        # Mostra os diretórios de teste/relatórios por base
        find "${results_dir}" -maxdepth 4 -type d -name "test_results" 2>/dev/null | while read -r d; do
            echo "   📂 ${d}"
            ls -1 "${d}" 2>/dev/null | sed 's/^/      - /'
        done
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
echo "🧪 Bases de teste (cada uma gera relatório SEPARADO):"
echo "  - ${TEST1_NAME}"
echo "  - ${TEST2_NAME}"
echo "  - ${TEST3_NAME}"
echo ""
echo "📁 Estrutura esperada por modelo:"
echo "   <results_dir>/<MODELO>_${TIMESTAMP}/"
echo "     ├── saved_models/"
echo "     ├── run_0/ ... run_N/"
echo "     └── <EXPERIMENT>/test_results/"
echo "          ├── ${TEST1_NAME}/"
echo "          ├── ${TEST2_NAME}/"
echo "          ├── ${TEST3_NAME}/"
echo "          └── COMPARATIVO_ENTRE_BASES/"
echo ""

# Verificar diretórios
check_directories
CHECK_EXIT=$?

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
# Formato: "Nome_do_Modelo" "caminho/do/script.py" batch_size learning_rate
# ============================================

MODELS=(
    "EfficientNetB0_UNet_g_enhanced" "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/five_efficientNet_cinza.py" ${BATCH_EFFICIENTNET} ${LR_EFFICIENTNET}
    "ResNet101_UNet_g_enhanced"      "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/five_resnet101_cinza.py"   ${BATCH_RESNET}       ${LR_RESNET}
    "SwinUNet_g_enhanced"            "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/five_swin_cinza.py"        ${BATCH_SWIN}         ${LR_SWIN}
    "VGG19_UNet_g_enhanced"          "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/five_vgg19_cinza.py"       ${BATCH_VGG}          ${LR_VGG}
    "ViTUNet_g_enhanced"             "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/five_vit_cinza.py"         ${BATCH_VIT}          ${LR_VIT}
    "MobileNetV2_UNet_g_enhanced"    "/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_pre_img_cinza/five_mobileNetV2net_cinza.py" ${BATCH_MOBILENET} ${LR_MOBILENET}
)

# Verificar scripts
echo ""
echo "🔍 Verificando scripts dos modelos (GRAYSCALE)..."
for ((i=0; i<${#MODELS[@]}; i+=4)); do
    SCRIPT_PATH=${MODELS[$((i+1))]}
    MODEL_NAME=${MODELS[$i]}
    if [ -f "${SCRIPT_PATH}" ]; then
        echo "  ✅ ${MODEL_NAME}"
    else
        echo "  ❌ ${MODEL_NAME} - Script NÃO ENCONTRADO:"
        echo "     ${SCRIPT_PATH}"
        DIR=$(dirname "${SCRIPT_PATH}")
        echo "     🔍 Arquivos similares em ${DIR}:"
        ls -la "${DIR}" 2>/dev/null | grep -i "cinza\|grayscale" | awk '{print "        - " $9}'
    fi
done
echo ""

TOTAL_MODELOS=$((${#MODELS[@]} / 4))
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
echo "  ❌ Modelos com falha:   ${FALHAS}/${TOTAL_MODELOS}"
echo ""

if [ ${FALHAS} -gt 0 ]; then
    echo "⚠️  Modelos que falharam:"
    for model in "${MODELOS_FALHAS[@]}"; do
        echo "    - ${model}"
    done
    echo ""
    echo "📁 Para verificar os logs:"
    echo "   cd ${BASE_RESULTS_DIR}"
    echo "   grep -r 'ERROR\\|Traceback' */execution.log"
fi

echo "📁 Resultados salvos em: ${BASE_RESULTS_DIR}/"
echo "🕐 Fim: $(date)"
echo ""

# ============================================
# LISTAGEM DE RELATÓRIOS GERADOS POR BASE
# ============================================
echo "📂 Estrutura de relatórios por modelo e base:"
for ((i=0; i<${#MODELS[@]}; i+=4)); do
    MODEL_NAME=${MODELS[$i]}
    MODEL_DIR="${BASE_RESULTS_DIR}/${MODEL_NAME}_${TIMESTAMP}"
    if [ -d "${MODEL_DIR}" ]; then
        echo ""
        echo "  🔹 ${MODEL_NAME}"
        # Procura por diretórios test_results dentro do experimento
        find "${MODEL_DIR}" -maxdepth 4 -type d -name "test_results" 2>/dev/null | while read -r tr; do
            echo "     📁 $(echo "${tr}" | sed "s|${MODEL_DIR}/||")"
            for base in "${tr}"/*/; do
                [ -d "${base}" ] || continue
                bname=$(basename "${base}")
                # Ignora o comparativo (mostra depois)
                if [ "${bname}" = "COMPARATIVO_ENTRE_BASES" ]; then
                    continue
                fi
                n_files=$(ls -1 "${base}" 2>/dev/null | wc -l)
                echo "        - ${bname}  (${n_files} arquivos)"
            done
            # Mostra comparativo
            comp="${tr}/COMPARATIVO_ENTRE_BASES"
            if [ -d "${comp}" ]; then
                echo "        - COMPARATIVO_ENTRE_BASES"
                ls -1 "${comp}" 2>/dev/null | sed 's/^/            * /'
            fi
        done
    fi
done

echo ""
echo "======================================================================"
if [ ${FALHAS} -eq 0 ]; then
    echo "🎉 TODOS OS MODELOS FORAM EXECUTADOS COM SUCESSO!"
    echo ""
    echo "📁 Para navegar nos resultados:"
    echo "   cd ${BASE_RESULTS_DIR}"
    echo "   # Cada modelo tem sua pasta; dentro de cada experimento:"
    echo "   #   <EXP>/test_results/<BASE>/REPORT.txt          (relatório individual)"
    echo "   #   <EXP>/test_results/COMPARATIVO_ENTRE_BASES/   (comparativo geral)"
    exit 0
else
    echo "⚠️  ALGUNS MODELOS FALHARAM. Verifique os logs para mais detalhes."
    exit 1
fi
