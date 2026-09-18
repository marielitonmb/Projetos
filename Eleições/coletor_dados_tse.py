# -*- coding: utf-8 -*-
"""
Script utilitário para Apoio à Coleta de Dados do TSE (Eleições 2026).
Fornece orientações, links diretos e estrutura de carga de dados abertos.
"""

import sys
from pathlib import Path
import pandas as pd

def listar_arquivos_propostas(diretorio_categoria):
    """Lista todos os arquivos PDF presentes no diretório informado."""
    dir_path = Path(diretorio_categoria)
    if not dir_path.exists():
        print(f"Diretório {dir_path} não encontrado.")
        return []
    
    arquivos = list(dir_path.glob("*.pdf"))
    print(f"\nEncontrados {len(arquivos)} arquivos em {dir_path.name}:")
    for a in arquivos:
        print(f"  - {a.name}")
    return arquivos

def carregar_amostra_dados_tse(caminho_csv):
    """Carrega dataset cadastral ou patrimonial do TSE."""
    csv_path = Path(caminho_csv)
    if not csv_path.exists():
        print(f"Arquivo CSV não encontrado: {csv_path}")
        return None
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    from extrator_propostas import detect_project_paths
    paths = detect_project_paths()
    print("=== DIRETÓRIOS DO PROJETO ELEIÇÕES 2026 ===")
    for k, v in paths.items():
        print(f"{k.upper():12}: {v}")
        
    listar_arquivos_propostas(paths['presidencia'])
    listar_arquivos_propostas(paths['governo_pb'])
