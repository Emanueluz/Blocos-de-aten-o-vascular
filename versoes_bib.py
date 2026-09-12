#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import platform
import subprocess
import importlib

# Lista de pacotes a verificar no formato (nome_do_import, nome_do_pacote_para_pip, atributo_de_versão)
packages = [
    ("torch", "torch", "__version__"),
    ("torchvision", "torchvision", "__version__"),
    ("cv2", "opencv-python", "__version__"),
    ("numpy", "numpy", "__version__"),
    ("tqdm", "tqdm", "__version__"),
    ("matplotlib", "matplotlib", "__version__"),
    ("sklearn", "scikit-learn", "__version__"),
    ("pandas", "pandas", "__version__"),
]

# Módulos built-in (sem versão específica)
builtins = ["os", "glob", "datetime", "json", "warnings", "time"]

def get_package_version(module_name, package_name, version_attr):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, version_attr, None)
        if version is None:
            # Tenta obter via pkg_resources se __version__ não existir
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
            except:
                version = "Desconhecida"
        return version
    except ImportError:
        return "❌ Não instalado"
    except Exception as e:
        return f"Erro: {e}"

def main():
    print("=" * 60)
    print("  INFORMAÇÕES DO SISTEMA E AMBIENTE PYTHON")
    print("=" * 60)
    print(f"Python versão: {sys.version}")
    print(f"Python executável: {sys.executable}")
    print(f"Plataforma: {platform.platform()}")
    print(f"Sistema operacional: {platform.system()} {platform.release()}")
    print("=" * 60)
    print("\nVERSÕES DOS PACOTES INSTALADOS:\n")

    # Verifica pacotes de terceiros
    for module_name, package_name, version_attr in packages:
        version = get_package_version(module_name, package_name, version_attr)
        print(f"  {module_name:15} : {version}")

    print("\nMÓDULOS BUILT-IN (sem versão específica):")
    for mod in builtins:
        try:
            importlib.import_module(mod)
            print(f"  {mod:15} : ✅ Disponível (built-in)")
        except ImportError:
            print(f"  {mod:15} : ❌ Não encontrado (improvável)")

    print("\n" + "=" * 60)
    print("  VERIFICAÇÃO CONCLUÍDA")
    print("=" * 60)

if __name__ == "__main__":
    main()