#!/bin/bash
# run_models.sh - Executa scripts com argumentos

# Configurações
BASE_DIR="/home/emanuel/Documentos/mestrado/treino dos modelos/modelos_img_originais/"
MODELOS=(
    "five_efficientNetB0UNet.py"
    "MobileNetV2UNet.py"
    "five_resnet101.py"
    "five_swin.py"
    "five_vgg19.py"
    "five_vit.py"
)

# Função para executar um script
run_script() {
    local script=$1
    echo "========================================="
    echo "Executando: $script"
    echo "Início: $(date)"
    echo "========================================="
    
    cd "$BASE_DIR"
    python3 "$script"
    
    if [ $? -eq 0 ]; then
        echo "✅ $script executado com sucesso!"
    else
        echo "❌ $script falhou!"
    fi
    echo ""
}

# Menu de seleção
echo "========================================="
echo "SELECIONE OS MODELOS PARA EXECUTAR"
echo "========================================="
echo "1 - Todos os modelos"
echo "2 - EfficientNetB0_UNet"
echo "3 - MobileNetV2UNet"
echo "4 - ResNet101_UNet"
echo "5 - SwinUNet"
echo "6 - VGG19_UNet"
echo "7 - ViTUNet"
echo "========================================="
read -p "Opção: " opcao

case $opcao in
    1)
        for modelo in "${MODELOS[@]}"; do
            run_script "$modelo"
        done
        ;;
    2)
        run_script "EfficientNetB0_UNet.py"
        ;;
    3)
        run_script "MobileNetV2UNet.py"
        ;;
    4)
        run_script "ResNet101_UNet.py"
        ;;
    5)
        run_script "SwinUNet.py"
        ;;
    6)
        run_script "VGG19_UNet.py"
        ;;
    7)
        run_script "ViTUNet.py"
        ;;
    *)
        echo "Opção inválida!"
        exit 1
        ;;
esac

echo "========================================="
echo "FINALIZADO EM: $(date)"
echo "========================================="