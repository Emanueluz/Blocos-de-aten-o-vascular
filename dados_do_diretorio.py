#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para analisar e documentar a organização dos scripts no diretório
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import csv

# ============================================
# CORES PARA OUTPUT (ANSI)
# ============================================
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


# ============================================
# FUNÇÕES DE ANÁLISE
# ============================================

def analyze_python_script(filepath):
    """Analisa um script Python e retorna suas métricas"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
                lines = content.split('\n')
        except:
            return None
    
    filename = os.path.basename(filepath)
    result = {
        'filename': filename,
        'path': filepath,
        'lines': len(lines),
        'size_kb': os.path.getsize(filepath) / 1024,
        'has_shebang': content.startswith('#!') and 'python' in content[:50].lower(),
        'has_docstring': '"""' in content[:500],
        'has_main': 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content,
        'has_argparse': 'argparse' in content,
        'classes': 0,
        'functions': 0,
        'imports': 0,
        'model_type': None,
        'mode': None,
        'import_list': []
    }
    
    # Classes
    result['classes'] = len(re.findall(r'^class\s+(\w+)', content, re.MULTILINE))
    
    # Funções
    result['functions'] = len(re.findall(r'^def\s+(\w+)', content, re.MULTILINE))
    
    # Imports
    imports = re.findall(r'^import\s+(\w+)', content, re.MULTILINE)
    from_imports = re.findall(r'^from\s+(\w+)', content, re.MULTILINE)
    result['imports'] = len(imports) + len(from_imports)
    result['import_list'] = imports + from_imports
    
    # Model type
    model_patterns = {
        'EfficientNet': r'efficientnet',
        'ResNet101': r'resnet101',
        'Swin': r'swin',
        'VGG19': r'vgg19',
        'ViT': r'vit',
        'MobileNetV2': r'mobilenetv2|mobilenet_v2'
    }
    
    for model, pattern in model_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            result['model_type'] = model
            break
    
    # Mode (RGB vs Grayscale)
    if re.search(r'grayscale|cinza', content, re.IGNORECASE):
        result['mode'] = 'Grayscale'
    elif re.search(r'rgb|color', content, re.IGNORECASE):
        result['mode'] = 'RGB'
    
    return result


def analyze_shell_script(filepath):
    """Analisa um script Shell e retorna suas métricas"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except:
        return None
    
    filename = os.path.basename(filepath)
    result = {
        'filename': filename,
        'path': filepath,
        'lines': len(lines),
        'size_kb': os.path.getsize(filepath) / 1024,
        'has_shebang': content.startswith('#!') and 'bash' in content[:50].lower(),
        'functions': 0,
        'models': []
    }
    
    # Funções Bash
    funcs = re.findall(r'^[[:space:]]*([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*\(\)', content, re.MULTILINE)
    result['functions'] = len(funcs)
    
    # Modelos detectados
    model_patterns = ['efficientnet', 'resnet', 'swin', 'vgg', 'vit', 'mobilenet']
    for pattern in model_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            result['models'].append(pattern.capitalize())
    
    return result


def analyze_config_file(filepath):
    """Analisa arquivos de configuração"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except:
        return None
    
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    
    result = {
        'filename': filename,
        'path': filepath,
        'lines': len(lines),
        'size_kb': os.path.getsize(filepath) / 1024,
        'type': ext[1:] if ext else 'unknown',
        'content_preview': content[:200]
    }
    
    return result


# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main():
    """Função principal do script"""
    
    # Diretório atual
    current_dir = os.getcwd()
    
    print(f"{Colors.BLUE}========================================={Colors.NC}")
    print(f"{Colors.BLUE}📊 ANALISANDO SCRIPTS DO DIRETÓRIO{Colors.NC}")
    print(f"{Colors.BLUE}========================================={Colors.NC}")
    print()
    print(f"📂 Diretório atual: {current_dir}")
    print()
    
    # Listar arquivos
    files = os.listdir(current_dir)
    
    python_files = [f for f in files if f.endswith('.py')]
    shell_files = [f for f in files if f.endswith('.sh')]
    config_files = [f for f in files if any(f.endswith(ext) for ext in ['.yml', '.yaml', '.json', '.txt', '.cfg', '.ini'])]
    
    all_results = {
        'python': [],
        'shell': [],
        'config': [],
        'summary': {}
    }
    
    # ============================================
    # ANALISAR SCRIPTS PYTHON
    # ============================================
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    print(f"{Colors.BLUE}#  SCRIPTS PYTHON{Colors.NC}")
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    
    if python_files:
        for file in sorted(python_files):
            filepath = os.path.join(current_dir, file)
            result = analyze_python_script(filepath)
            if result:
                all_results['python'].append(result)
                
                print()
                print(f"{Colors.YELLOW}📄 {result['filename']}{Colors.NC}")
                print(f"{Colors.BLUE}─────────────────────────────────────────{Colors.NC}")
                
                # Shebang
                status = f"{Colors.GREEN}✅{Colors.NC}" if result['has_shebang'] else f"{Colors.YELLOW}⚠️{Colors.NC}"
                print(f"  {status} Shebang: {'Sim' if result['has_shebang'] else 'Não'}")
                
                # Docstring
                status = f"{Colors.GREEN}✅{Colors.NC}" if result['has_docstring'] else f"{Colors.YELLOW}⚠️{Colors.NC}"
                print(f"  {status} Docstring: {'Sim' if result['has_docstring'] else 'Não'}")
                
                # Modelo
                if result['model_type']:
                    print(f"  {Colors.GREEN}✅{Colors.NC} Modelo: {result['model_type']}")
                else:
                    print(f"  {Colors.YELLOW}⚠️{Colors.NC}  Modelo não identificado")
                
                # Modo
                if result['mode']:
                    print(f"  {Colors.GREEN}✅{Colors.NC} Modo: {result['mode']}")
                
                # Métricas
                print(f"  📏 Linhas: {result['lines']}")
                print(f"  💾 Tamanho: {result['size_kb']:.1f} KB")
                print(f"  🏗️  Classes: {result['classes']}")
                print(f"  🔧 Funções: {result['functions']}")
                print(f"  📦 Importações: {result['imports']}")
                
                # Main
                status = f"{Colors.GREEN}✅{Colors.NC}" if result['has_main'] else f"{Colors.YELLOW}⚠️{Colors.NC}"
                print(f"  {status} Main: {'Sim' if result['has_main'] else 'Não'}")
                
                # Argparse
                if result['has_argparse']:
                    print(f"  {Colors.GREEN}✅{Colors.NC} Argumentos via CLI: Sim")
                
                print(f"{Colors.BLUE}─────────────────────────────────────────{Colors.NC}")
    else:
        print(f"{Colors.RED}❌ Nenhum arquivo Python encontrado{Colors.NC}")
    
    # ============================================
    # ANALISAR SCRIPTS SHELL
    # ============================================
    print()
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    print(f"{Colors.BLUE}#  SCRIPTS SHELL{Colors.NC}")
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    
    if shell_files:
        for file in sorted(shell_files):
            filepath = os.path.join(current_dir, file)
            result = analyze_shell_script(filepath)
            if result:
                all_results['shell'].append(result)
                
                print()
                print(f"{Colors.YELLOW}📄 {result['filename']}{Colors.NC}")
                print(f"{Colors.BLUE}─────────────────────────────────────────{Colors.NC}")
                
                # Shebang
                status = f"{Colors.GREEN}✅{Colors.NC}" if result['has_shebang'] else f"{Colors.YELLOW}⚠️{Colors.NC}"
                print(f"  {status} Shebang: {'Sim' if result['has_shebang'] else 'Não'}")
                
                # Métricas
                print(f"  📏 Linhas: {result['lines']}")
                print(f"  💾 Tamanho: {result['size_kb']:.1f} KB")
                print(f"  🔧 Funções: {result['functions']}")
                
                # Modelos
                if result['models']:
                    print(f"  🧠 Modelos: {', '.join(result['models'])}")
                
                print(f"{Colors.BLUE}─────────────────────────────────────────{Colors.NC}")
    else:
        print(f"{Colors.RED}❌ Nenhum arquivo Shell encontrado{Colors.NC}")
    
    # ============================================
    # ANALISAR ARQUIVOS DE CONFIGURAÇÃO
    # ============================================
    print()
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    print(f"{Colors.BLUE}#  ARQUIVOS DE CONFIGURAÇÃO{Colors.NC}")
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    
    if config_files:
        for file in sorted(config_files):
            filepath = os.path.join(current_dir, file)
            result = analyze_config_file(filepath)
            if result:
                all_results['config'].append(result)
                
                print()
                print(f"{Colors.YELLOW}📄 {result['filename']}{Colors.NC}")
                print(f"{Colors.BLUE}─────────────────────────────────────────{Colors.NC}")
                print(f"  📏 Linhas: {result['lines']}")
                print(f"  💾 Tamanho: {result['size_kb']:.1f} KB")
                print(f"  📋 Tipo: {result['type']}")
                print(f"{Colors.BLUE}─────────────────────────────────────────{Colors.NC}")
    else:
        print(f"{Colors.RED}❌ Nenhum arquivo de configuração encontrado{Colors.NC}")
    
    # ============================================
    # RELATÓRIO RESUMIDO
    # ============================================
    print()
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    print(f"{Colors.BLUE}#  RELATÓRIO RESUMIDO{Colors.NC}")
    print(f"{Colors.BLUE}########################################{Colors.NC}")
    print()
    
    # Contagem
    print(f"📊 Estatísticas:")
    print(f"  🐍 Python scripts: {len(python_files)}")
    print(f"  📄 Shell scripts: {len(shell_files)}")
    print(f"  ⚙️  Configurações: {len(config_files)}")
    print(f"  📂 Total de arquivos: {len(files)}")
    print()
    
    # Modelos encontrados
    model_names = set()
    for result in all_results['python']:
        if result['model_type']:
            model_names.add(result['model_type'])
    
    print(f"🧠 Modelos detectados:")
    if model_names:
        for model in sorted(model_names):
            print(f"  {Colors.GREEN}✅{Colors.NC} {model}")
    else:
        print(f"  {Colors.YELLOW}⚠️{Colors.NC}  Nenhum modelo identificado")
    
    # Modos encontrados
    modes = set()
    for result in all_results['python']:
        if result['mode']:
            modes.add(result['mode'])
    
    if modes:
        print()
        print(f"🎨 Modos disponíveis:")
        for mode in sorted(modes):
            count = sum(1 for r in all_results['python'] if r['mode'] == mode)
            print(f"  {Colors.GREEN}✅{Colors.NC} {mode}: {count} script(s)")
    
    # Scripts com CLI
    cli_scripts = [r['filename'] for r in all_results['python'] if r['has_argparse']]
    if cli_scripts:
        print()
        print(f"💻 Scripts com CLI:")
        for script in sorted(cli_scripts):
            print(f"  - {script}")
    
    print()
    print(f"{Colors.BLUE}========================================={Colors.NC}")
    print(f"{Colors.GREEN}✅ Análise concluída!{Colors.NC}")
    print(f"{Colors.BLUE}========================================={Colors.NC}")
    
    # ============================================
    # PERGUNTAR SOBRE RELATÓRIO
    # ============================================
    print()
    resp = input(f"{Colors.YELLOW}📊 Gerar relatório detalhado? (s/N): {Colors.NC}")
    
    if resp.lower() == 's':
        generate_report(all_results, current_dir)


def generate_report(results, directory):
    """Gera relatórios detalhados em diferentes formatos"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ============================================
    # RELATÓRIO EM JSON
    # ============================================
    json_file = f"script_analysis_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"{Colors.GREEN}✅ Relatório JSON salvo: {json_file}{Colors.NC}")
    
    # ============================================
    # RELATÓRIO EM CSV
    # ============================================
    if results['python']:
        csv_file = f"script_analysis_python_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Nome', 'Linhas', 'Tamanho(KB)', 'Classes', 
                'Funções', 'Importações', 'Modelo', 'Modo', 
                'Shebang', 'Docstring', 'Main', 'CLI'
            ])
            for r in results['python']:
                writer.writerow([
                    r['filename'],
                    r['lines'],
                    f"{r['size_kb']:.1f}",
                    r['classes'],
                    r['functions'],
                    r['imports'],
                    r['model_type'] or 'N/A',
                    r['mode'] or 'N/A',
                    'Sim' if r['has_shebang'] else 'Não',
                    'Sim' if r['has_docstring'] else 'Não',
                    'Sim' if r['has_main'] else 'Não',
                    'Sim' if r['has_argparse'] else 'Não'
                ])
        print(f"{Colors.GREEN}✅ Relatório CSV salvo: {csv_file}{Colors.NC}")
    
    # ============================================
    # RELATÓRIO EM MARKDOWN
    # ============================================
    md_file = f"script_analysis_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Análise de Scripts - {directory}\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        # Python scripts
        f.write("## Scripts Python\n\n")
        f.write("| Nome | Linhas | Tamanho | Classes | Funções | Modelo | Modo |\n")
        f.write("|------|--------|---------|---------|---------|--------|------|\n")
        for r in sorted(results['python'], key=lambda x: x['filename']):
            f.write(f"| {r['filename']} | {r['lines']} | {r['size_kb']:.1f}KB | {r['classes']} | {r['functions']} | {r['model_type'] or 'N/A'} | {r['mode'] or 'N/A'} |\n")
        
        # Shell scripts
        f.write("\n## Scripts Shell\n\n")
        f.write("| Nome | Linhas | Tamanho | Funções | Modelos |\n")
        f.write("|------|--------|---------|---------|---------|\n")
        for r in sorted(results['shell'], key=lambda x: x['filename']):
            f.write(f"| {r['filename']} | {r['lines']} | {r['size_kb']:.1f}KB | {r['functions']} | {', '.join(r['models']) or 'N/A'} |\n")
        
        # Config files
        if results['config']:
            f.write("\n## Arquivos de Configuração\n\n")
            f.write("| Nome | Linhas | Tamanho | Tipo |\n")
            f.write("|------|--------|---------|------|\n")
            for r in sorted(results['config'], key=lambda x: x['filename']):
                f.write(f"| {r['filename']} | {r['lines']} | {r['size_kb']:.1f}KB | {r['type']} |\n")
        
        # Summary
        f.write("\n## Resumo\n\n")
        f.write(f"- **Python scripts:** {len(results['python'])}\n")
        f.write(f"- **Shell scripts:** {len(results['shell'])}\n")
        f.write(f"- **Config files:** {len(results['config'])}\n")
        
        # Models
        model_names = sorted(set(r['model_type'] for r in results['python'] if r['model_type']))
        if model_names:
            f.write(f"- **Modelos encontrados:** {', '.join(model_names)}\n")
    
    print(f"{Colors.GREEN}✅ Relatório Markdown salvo: {md_file}{Colors.NC}")
    
    print()
    print(f"{Colors.GREEN}✅ Todos os relatórios gerados com sucesso!{Colors.NC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Execução interrompida pelo usuário{Colors.NC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Erro: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)