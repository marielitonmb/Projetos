# -*- coding: utf-8 -*-
"""
Módulo de Extração, Limpeza e Visualização Analítica - Eleições 2026.
Fornece funções para gráficos isolados por candidato, detecção inteligente de tópicos
(incluindo Privatização, Feminicídio, Minorias, Presídios, LGBTQIA+) e prestação de contas.
"""

import os
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import pypdf

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Garante download silencioso dos recursos do NLTK
for resource in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


def detect_project_paths():
    """
    Detecta automaticamente caminhos do projeto para Local Windows e Google Colab.
    Suporta tanto a estrutura com subpastas (Arquivos/Presidencia) quanto
    arquivos carregados diretamente na raiz do Colab (/content).
    """
    colab_base = Path('/content')
    if colab_base.exists():
        base_dir = colab_base
    else:
        current = Path.cwd()
        if (current / 'Arquivos').exists() or list(current.glob('*.pdf')):
            base_dir = current
        elif (current.parent / 'Arquivos').exists() or list(current.parent.glob('*.pdf')):
            base_dir = current.parent
        else:
            base_dir = Path(r'D:/Documents/Meus-Projetos/Análise-Eleições-2026')
    
    # Detecção inteligente da pasta Presidência
    if (base_dir / 'Arquivos' / 'Presidencia').exists() and list((base_dir / 'Arquivos' / 'Presidencia').glob('*.pdf')):
        dir_pres = base_dir / 'Arquivos' / 'Presidencia'
    elif (base_dir / 'Presidencia').exists() and list((base_dir / 'Presidencia').glob('*.pdf')):
        dir_pres = base_dir / 'Presidencia'
    else:
        dir_pres = base_dir

    # Detecção inteligente da pasta Governo PB
    if (base_dir / 'Arquivos' / 'Governo_PB').exists() and list((base_dir / 'Arquivos' / 'Governo_PB').glob('*.pdf')):
        dir_pb = base_dir / 'Arquivos' / 'Governo_PB'
    elif (base_dir / 'Governo_PB').exists() and list((base_dir / 'Governo_PB').glob('*.pdf')):
        dir_pb = base_dir / 'Governo_PB'
    else:
        dir_pb = base_dir

    # Detecção inteligente da pasta Dados TSE
    if (base_dir / 'Arquivos' / 'Dados_TSE').exists() and list((base_dir / 'Arquivos' / 'Dados_TSE').glob('*.csv')):
        dir_dados = base_dir / 'Arquivos' / 'Dados_TSE'
    elif (base_dir / 'Dados_TSE').exists() and list((base_dir / 'Dados_TSE').glob('*.csv')):
        dir_dados = base_dir / 'Dados_TSE'
    else:
        dir_dados = base_dir

    return {
        'base': base_dir,
        'arquivos': base_dir / 'Arquivos' if (base_dir / 'Arquivos').exists() else base_dir,
        'presidencia': dir_pres,
        'governo_pb': dir_pb,
        'dados_tse': dir_dados,
        'codigo': base_dir / 'Código' if (base_dir / 'Código').exists() else base_dir
    }


def format_candidate_label(filename_or_name):
    """
    Padroniza a identificação do candidato estritamente no formato 'Nome (Partido)'.
    Suporta tanto o formato 'Nome - PARTIDO.pdf' quanto arquivos históricos.
    """
    name = str(filename_or_name).replace("Proposta-", "").replace(".pdf", "").strip()
    
    # Se já estiver com parênteses, retorna direto
    if "(" in name and ")" in name:
        return name
        
    # Formato do usuário: 'Nome - PARTIDO'
    if " - " in name:
        parts = name.split(" - ")
        nome_candidato = parts[0].strip()
        partido = parts[1].strip()
        return f"{nome_candidato} ({partido.upper()})"

    clean_key = name.lower().replace("-", " ").replace("_", " ").strip()
    clean_key = " ".join(clean_key.split())
    
    mapping = {
        "lula": "Lula (PT)",
        "lula pt": "Lula (PT)",
        "flavio bolsonaro": "Flavio Bolsonaro (PL)",
        "romeu zema": "Romeu Zema (NOVO)",
        "ronaldo caiado": "Ronaldo Caiado (PSD)",
        "augusto cury": "Augusto Cury (AVANTE)",
        "renan santos": "Renan Santos (MISSÃO)",
        "felipe d'avila": "Felipe D'Avila (NOVO)",
        "simone tebet": "Simone Tebet (MDB)",
        "cicero lucena": "Cícero Lucena (MDB)",
        "efraim filho": "Efraim Filho (PL)",
        "lucas ribeiro": "Lucas Ribeiro (PP)",
        "marcelo queiroga": "Marcelo Queiroga (PL)",
        "ruy carneiro": "Ruy Carneiro (PODEMOS)"
    }
    
    if clean_key in mapping:
        return mapping[clean_key]
        
    for k, v in mapping.items():
        if k in clean_key or clean_key in k:
            return v
            
    return f"{name.title()} (Candidato)"


def get_pdf_page_count(filepath):
    """Retorna a contagem total de páginas de um arquivo PDF."""
    try:
        reader = pypdf.PdfReader(str(filepath))
        return len(reader.pages)
    except Exception:
        return 0


def read_file_pdf(filepath):
    """Extrai texto do PDF com fallback de pypdf para pdfplumber."""
    filepath = Path(filepath)
    if not filepath.is_file():
        print(f"[AVISO] Arquivo não encontrado: {filepath.name}")
        return ""
    
    text = ""
    try:
        reader = pypdf.PdfReader(str(filepath))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception:
        pass

    if not text.strip() and pdfplumber is not None:
        try:
            with pdfplumber.open(str(filepath)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except Exception:
            pass

    if not text.strip():
        print(f"[ALERTA] Documento escaneado/imagem sem texto extraível: {filepath.name}")
        
    return text


def extract_keywords(text, ignore_words=None, min_word_length=3, ignore_numbers=True, ignore_case=True):
    """Tokeniza o texto e extrai palavras-chave filtradas de stopwords e ruídos eleitorais."""
    if not text:
        return []
    
    tokens = word_tokenize(text)
    punctuations = [
        '(', ')', ';', ':', '[', ']', ',', '.', '--', '-', '#', '!', '*', '"', '%',
        '/', '\\', '?', '>', '<', '=', '+', '_', '{', '}', '@', '$', '§', '“', '”', '’', '‘'
    ]
    
    try:
        stop_words = set(stopwords.words('portuguese'))
    except Exception:
        stop_words = set()

    electoral_stopwords = {
        'governo', 'plano', 'proposta', 'propostas', 'candidato', 'candidatos',
        'programa', 'gestão', 'desenvolvimento', 'ação', 'ações', 'ano', 'anos',
        'cada', 'sobre', 'ainda', 'forma', 'todos', 'todas', 'país', 'estado',
        'município', 'nacional', 'público', 'pública', 'públicas', 'públicos',
        'social', 'sociais', 'novo', 'nova', 'novos', 'novas', 'projeto', 'projetos',
        'sistema', 'setor', 'área', 'áreas', 'população', 'pessoas', 'geral', 'bem',
        'assim', 'onde', 'além', 'através', 'fazer', 'garantir', 'promover', 'ampliar',
        'criar', 'fortalecer', 'melhorar', 'deve', 'devem', 'ser', 'ter', 'estar',
        'brasil', 'paraíba', 'brasileiro', 'brasileira', 'paraibano', 'paraibana'
    }
    
    all_stop = stop_words.union(electoral_stopwords)
    if ignore_words:
        all_stop.update([w.lower() for w in ignore_words])
    
    keywords = []
    for word in tokens:
        w_clean = word.strip()
        w_lower = w_clean.lower()
        if (w_clean not in punctuations and 
            w_lower not in all_stop and 
            len(w_clean) >= min_word_length):
            
            if ignore_numbers and (w_clean.isdigit() or any(c.isdigit() for c in w_clean)):
                continue
            
            keywords.append(w_lower if ignore_case else w_clean)
            
    return keywords


def count_term_with_variations(text, keywords, target_topic):
    """
    Contabiliza menções de um tema considerando flexões morfológicas,
    sinônimos e termos compostos.
    Especialmente projetado para os 5 tópicos solicitados:
    - Privatização
    - Feminicídio
    - Minorias
    - Presídios
    - LGBTQIA+
    """
    topic_clean = target_topic.lower().strip()
    text_lower = text.lower() if text else ""
    
    term_variants = {
        'privatização': ['privatização', 'privatizações', 'privatizar', 'desestatização', 'desestatizações'],
        'feminicídio': ['feminicídio', 'feminicídios', 'violência contra a mulher'],
        'minorias': ['minorias', 'minoria', 'grupos minoritários', 'populações vulneráveis'],
        'presídios': ['presídios', 'presídio', 'prisão', 'prisões', 'penitenciária', 'penitenciárias', 'sistema prisional', 'carcerário', 'carcerária'],
        'lgbtqia+': ['lgbt', 'lgbtq', 'lgbtqia', 'lgbtqia+', 'lgbti', 'homofobia', 'transfobia', 'diversidade sexual']
    }
    
    variants = term_variants.get(topic_clean, [topic_clean])
    total_count = 0
    
    for v in variants:
        # Se for termo composto (ex: 'violência contra a mulher' ou 'sistema prisional')
        if " " in v or "+" in v:
            # Conta no texto corrido usando regex
            pattern = re.escape(v)
            matches = re.findall(pattern, text_lower)
            total_count += len(matches)
        else:
            # Conta a partir da lista de keywords
            total_count += keywords.count(v)
            
    return total_count


# ==============================================================================
# FUNÇÕES DE VISUALIZAÇÃO ESTRITAMENTE ISOLADAS POR CANDIDATO
# ==============================================================================

def plot_candidate_wordcloud_isolated(keywords, candidate_label):
    """Plota a nuvem de palavras de um único candidato isoladamente."""
    if not keywords:
        print(f"[AVISO] Sem palavras-chave para {candidate_label}.")
        return
    
    word_freq = Counter(keywords)
    wc = WordCloud(background_color='white', max_words=90, colormap='Dark2',
                   stopwords=STOPWORDS, max_font_size=240,
                   random_state=42, width=2200, height=1100).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(12, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f'Nuvem de Palavras - {candidate_label}', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_candidate_subjects_isolated(keywords, text, subjects, candidate_label, color='darkcyan', category_title='Eixos Temáticos'):
    """Plota gráfico de barras isolado para os eixos temáticos do candidato."""
    counts = [count_term_with_variations(text, keywords, s) for s in subjects]
    
    plt.figure(figsize=(14, 4.5))
    bars = plt.bar(subjects, counts, color=color, edgecolor='black', alpha=0.85)
    
    for i, count in enumerate(counts):
        plt.text(i, count + 0.3, str(count), ha='center', va='bottom',
                 fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
    plt.title(f'{category_title} - {candidate_label}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('ASSUNTOS PESQUISADOS', fontsize=10, fontweight='bold')
    plt.ylabel('OCORRÊNCIAS', fontsize=10, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_candidate_critical_topics_isolated(text, keywords, candidate_label):
    """
    Plota gráfico de barras isolado exclusivamente com os 5 temas solicitados:
    Privatização, Feminicídio, Minorias, Presídios e LGBTQIA+.
    """
    temas = ['Privatização', 'Feminicídio', 'Minorias', 'Presídios', 'LGBTQIA+']
    cores = ['#D9534F', '#AA66CC', '#33B5E5', '#F0AD4E', '#FF4444']
    
    counts = [count_term_with_variations(text, keywords, t) for t in temas]
    
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(temas, counts, color=cores, edgecolor='black', alpha=0.85)
    
    for i, count in enumerate(counts):
        plt.text(i, count + 0.2, str(count), ha='center', va='bottom',
                 fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDD0', alpha=0.9, edgecolor='gray'))
        
    plt.title(f'Temas Críticos Mapeados na Proposta - {candidate_label}', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('TOTAL DE CITAÇÕES / MENÇÕES', fontsize=10, fontweight='bold')
    plt.xlabel('TÓPICO OBRIGATÓRIO', fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_candidate_financial_sources_isolated(row_financeiro):
    """Plota gráfico em formato donut/pizza isolado com as fontes de receita do candidato."""
    nome = row_financeiro['nome_formatado']
    
    fontes = {
        'Fundo Eleitoral (FEFC)': row_financeiro['fundo_eleitoral'],
        'Fundo Partidário': row_financeiro['fundo_partidario'],
        'Doações Pessoas Físicas': row_financeiro['doacoes_pessoas_fisicas'],
        'Recursos Próprios': row_financeiro['recursos_proprios'],
        'Outros Recursos': row_financeiro.get('outros_recursos', 0.0)
    }
    
    # Filtra apenas fontes maiores que zero
    fontes_ativas = {k: v for k, v in fontes.items() if v > 0}
    
    if not fontes_ativas:
        print(f"[Info] Sem receitas declaradas registradas para {nome}.")
        return

    labels = list(fontes_ativas.keys())
    valores = list(fontes_ativas.values())
    total = sum(valores)
    
    cores = ['#4A90E2', '#50E3C2', '#F5A623', '#BD10E0', '#B8E986'][:len(valores)]
    
    plt.figure(figsize=(9, 5))
    wedges, texts, autotexts = plt.pie(
        valores, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=cores, pctdistance=0.75, explode=[0.04]*len(valores),
        textprops=dict(fontweight='bold', fontsize=9.5)
    )
    
    # Transforma em gráfico Donut
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title(f'Prestação de Contas: Origem dos Recursos - {nome}\nTotal Arrecadado: R$ {total:,.2f}'.replace(",", "X").replace(".", ",").replace("X", "."),
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_candidate_financial_execution_isolated(row_financeiro):
    """Plota gráfico de barras isolado da execução financeira (Limite x Receitas x Despesas)."""
    nome = row_financeiro['nome_formatado']
    
    categorias = ['Limite de Gastos', 'Receitas Arrecadadas', 'Despesas Contratadas', 'Despesas Pagas']
    valores = [
        row_financeiro['limite_gastos_1t'],
        row_financeiro['total_receitas'],
        row_financeiro['total_despesas_contratadas'],
        row_financeiro['total_despesas_pagas']
    ]
    cores = ['#6C757D', '#28A745', '#FFC107', '#17A2B8']
    
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(categorias, valores, color=cores, edgecolor='black', alpha=0.85)
    
    for bar in bars:
        h = bar.get_height()
        texto_valor = f"R$ {h/1e6:.2f}M" if h >= 1e6 else f"R$ {h/1e3:.0f}K"
        plt.text(bar.get_x() + bar.get_width()/2, h + (max(valores)*0.02),
                 texto_valor, ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    plt.title(f'Execução Orçamentária de Campanha - {nome}', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('VALOR DECLARADO NO TSE (R$)', fontweight='bold', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Formata eixo Y em milhões
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1e6:.1f}M' if x >= 1e6 else f'R$ {x/1e3:.0f}K'))
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_candidate_profile_isolated(row_perfil):
    """Exibe ficha demográfica, histórico partidário e patrimonial do candidato analisado."""
    nome = row_perfil['nome_formatado']
    bens_fmt = f"R$ {row_perfil.get('bens_declarados_reais', 0.0):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    historico = row_perfil.get('historico_partidos', row_perfil.get('partido', 'N/A'))
    
    print(f"\n{'='*80}")
    print(f"   FICHA CADASTRAL, HISTÓRICO PARTIDÁRIO E PATRIMÔNIO (TSE) - {nome.upper()}")
    print(f"{'='*80}")
    print(f"  Nome Completo:        {row_perfil.get('nome_completo', 'N/A')}")
    print(f"  Partido Atual:        {row_perfil.get('partido', 'N/A')}")
    print(f"  Histórico de Partidos: {historico}")
    print(f"  Gênero / Raça:        {row_perfil.get('genero', 'N/A')} | {row_perfil.get('cor_raca', 'N/A')}")
    print(f"  Idade / Escolaridade: {row_perfil.get('idade', 'N/A')} anos | {row_perfil.get('grau_instrucao', 'N/A')}")
    print(f"  Ocupação Declarada:   {row_perfil.get('ocupacao', 'N/A')}")
    print(f"  Bens Declarados:      {bens_fmt}")
    print(f"{'='*80}\n")
