#!/bin/bash

# ============================================
# SCRIPT PARA CRIAR AMBIENTE VIRTUAL PYTHON
# Com as versões específicas dos pacotes
# ============================================

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  CRIAÇÃO DE AMBIENTE VIRTUAL PYTHON${NC}"
echo -e "${BLUE}============================================================${NC}"

# Verificar se python3 está instalado
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 não encontrado!${NC}"
    echo "Instale com: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Mostrar versão do Python
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Python encontrado: ${PYTHON_VERSION}${NC}"

# ============================================
# CONFIGURAÇÕES
# ============================================

# Nome do ambiente virtual (pode ser alterado)
VENV_NAME="venv_segmentacao"

# Versões exatas dos pacotes
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
CUDA_VERSION="cu121"  # ou "cpu" se não tiver GPU

# Versões dos demais pacotes
OPENCV_VERSION="4.10.0.84"
NUMPY_VERSION="2.2.6"
TQDM_VERSION="4.66.5"
MATPLOTLIB_VERSION="3.9.2"
SKLEARN_VERSION="1.5.1"
PANDAS_VERSION="2.2.3"

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

print_step() {
    echo -e "\n${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ============================================
# CRIAÇÃO DO AMBIENTE VIRTUAL
# ============================================

print_step "Criando ambiente virtual '$VENV_NAME'..."

# Remover ambiente antigo se existir
if [ -d "$VENV_NAME" ]; then
    print_info "Ambiente '$VENV_NAME' já existe. Removendo..."
    rm -rf "$VENV_NAME"
fi

# Criar novo ambiente virtual
python3 -m venv "$VENV_NAME"

if [ ! -d "$VENV_NAME" ]; then
    print_error "Falha ao criar ambiente virtual!"
    exit 1
fi

print_success "Ambiente virtual criado com sucesso!"

# ============================================
# ATIVAR AMBIENTE VIRTUAL
# ============================================

print_step "Ativando ambiente virtual..."

# Ativar o ambiente virtual
source "$VENV_NAME/bin/activate"

if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Falha ao ativar ambiente virtual!"
    exit 1
fi

print_success "Ambiente virtual ativado!"

# ============================================
# ATUALIZAR PIP
# ============================================

print_step "Atualizando pip..."

pip install --upgrade pip setuptools wheel

print_success "Pip atualizado!"

# ============================================
# INSTALAR PYTORCH (COM CUDA)
# ============================================

print_step "Instalando PyTorch ${TORCH_VERSION}+${CUDA_VERSION}..."

# PyTorch com CUDA
pip install torch==${TORCH_VERSION}+${CUDA_VERSION} torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Verificar instalação
if python -c "import torch; import torchvision" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    print_success "PyTorch instalado: ${TORCH_VER}"
else
    print_error "Falha ao instalar PyTorch! Tentando versão CPU..."
    # Fallback para CPU
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu
fi

# ============================================
# INSTALAR DEMAIS PACOTES
# ============================================

print_step "Instalando OpenCV ${OPENCV_VERSION}..."

pip install opencv-python==${OPENCV_VERSION}

if python -c "import cv2" 2>/dev/null; then
    CV2_VER=$(python -c "import cv2; print(cv2.__version__)")
    print_success "OpenCV instalado: ${CV2_VER}"
else
    print_error "Falha ao instalar OpenCV!"
fi

print_step "Instalando NumPy ${NUMPY_VERSION}..."
pip install numpy==${NUMPY_VERSION}
print_success "NumPy instalado"

print_step "Instalando tqdm ${TQDM_VERSION}..."
pip install tqdm==${TQDM_VERSION}
print_success "tqdm instalado"

print_step "Instalando Matplotlib ${MATPLOTLIB_VERSION}..."
pip install matplotlib==${MATPLOTLIB_VERSION}
print_success "Matplotlib instalado"

print_step "Instalando Scikit-learn ${SKLEARN_VERSION}..."
pip install scikit-learn==${SKLEARN_VERSION}
print_success "Scikit-learn instalado"

print_step "Instalando Pandas ${PANDAS_VERSION}..."
pip install pandas==${PANDAS_VERSION}
print_success "Pandas instalado"

# ============================================
# INSTALAR DEPENDÊNCIAS ADICIONAIS
# ============================================

print_step "Instalando dependências adicionais..."

pip install pillow seaborn scipy

print_success "Dependências adicionais instaladas"

# ============================================
# VERIFICAR INSTALAÇÕES
# ============================================

print_step "Verificando todas as instalações..."

# Criar script de verificação
cat > check_versions.py << 'EOF'
#!/usr/bin/env python3

import sys
import platform

print("=" * 60)
print("  INFORMAÇÕES DO AMBIENTE VIRTUAL")
print("=" * 60)
print(f"Python versão: {sys.version}")
print(f"Python executável: {sys.executable}")
print(f"Plataforma: {platform.platform()}")
print(f"Sistema operacional: {platform.system()} {platform.release()}")
print("=" * 60)
print("\nVERSÕES DOS PACOTES INSTALADOS:\n")

packages = {
    "torch": "torch",
    "torchvision": "torchvision", 
    "cv2": "opencv-python",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "pandas": "pandas"
}

for import_name, package_name in packages.items():
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
            except:
                version = "Desconhecida"
        print(f"  {import_name:15} : {version}")
    except ImportError:
        print(f"  {import_name:15} : ❌ Não instalado")
    except Exception as e:
        print(f"  {import_name:15} : Erro: {e}")

print("\nMÓDULOS BUILT-IN (verificados):")
builtins = ["os", "glob", "datetime", "json", "warnings", "time"]
for mod in builtins:
    try:
        __import__(mod)
        print(f"  {mod:15} : ✅ Disponível")
    except ImportError:
        print(f"  {mod:15} : ❌ Não encontrado")

print("\n" + "=" * 60)
print("  VERIFICAÇÃO CONCLUÍDA")
print("=" * 60)
EOF

# Executar verificação
python check_versions.py

# ============================================
# CRIAR ARQUIVO DE REQUISITOS
# ============================================

print_step "Criando arquivo requirements.txt..."

pip freeze > requirements.txt

print_success "Requirements.txt criado!"

# ============================================
# MENSAGEM FINAL
# ============================================

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ AMBIENTE VIRTUAL CONFIGURADO COM SUCESSO!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "Para ativar o ambiente virtual:"
echo -e "${YELLOW}source $VENV_NAME/bin/activate${NC}"
echo ""
echo -e "Para desativar:"
echo -e "${YELLOW}deactivate${NC}"
echo ""
echo -e "Para verificar as instalações:"
echo -e "${YELLOW}python check_versions.py${NC}"
echo ""
echo -e "Arquivo de requisitos gerado: ${BLUE}requirements.txt${NC}"
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}🎉 AMBIENTE VIRTUAL PRONTO PARA USO!${NC}"
echo -e "${BLUE}============================================================${NC}"

# ============================================
# OFERECER ATIVAR AUTOMATICAMENTE
# ============================================

echo ""
read -p "Deseja ativar o ambiente virtual agora? (s/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo -e "${GREEN}Ativando ambiente virtual...${NC}"
    source "$VENV_NAME/bin/activate"
    echo -e "${GREEN}Ambiente ativado!${NC}"
    echo -e "Versão do Python: $(python --version)"
    echo -e "Pip: $(pip --version)"
fi