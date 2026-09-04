# RoBERTa-pt-br-juridico
a RoBERTa model trained on Brazilian legal data from the Ulysses Tesemo corpus

# JurisNER — Reconhecimento de Entidades Nomeadas no Domínio Jurídico Brasileiro

> Trabalho de Conclusão de Curso (PFC2) — Bacharelado em Ciências da Computação  
> Universidade Federal de Catalão (UFCAT) — 2025  
> **Autor:** Bruno Martins Costa  
> **Orientador:** Prof. Dr. Marcio de Souza Dias

---

## Sobre o Projeto

Este repositório contém o pipeline completo do **JurisNER**, um projeto que compara o desempenho de duas arquiteturas Transformer em tarefas de **Reconhecimento de Entidades Nomeadas (NER)** no domínio jurídico brasileiro:

| Modelo | HuggingFace |
|--------|-------------|
| **BERTimbau** | `neuralmind/bert-base-portuguese-cased` |
| **XLM-RoBERTa** | `xlm-roberta-base` |

A abordagem utiliza **Domain Adaptive Pretraining (DAPT)** — os modelos são submetidos a um pré-treinamento adicional com o corpus jurídico **Ulysses Tesemô** antes do fine-tuning de NER no dataset **LENER-Br**.

---

## Resultados

Avaliação no conjunto de teste do LENER-Br (F1-score):

| Entidade | BERTimbau | XLM-RoBERTa |
|---|---|---|
| **JURISPRUDÊNCIA** | — | **0.867** |
| **LEGISLAÇÃO** | — | **0.957** |
| **LOCAL** | — | 0.673 |
| **ORGANIZAÇÃO** | — | 0.857 |
| **PESSOA** | — | **0.944** |
| **TEMPO** | — | **0.960** |
| **F1 Geral (micro)** | **0.9027** | **0.9021** |
| **F1 Macro** | — | 0.876 |

> Os dois modelos apresentam desempenho competitivo e comparável ao estado da arte para NER jurídico em português, com vantagens complementares por categoria de entidade.

---

## Estrutura do Repositório

```
JurisNER/
├── tesemo_pipeline.py        # Limpeza e deduplicação do corpus Tesemô
├── jurisroberta_pipeline.py  # Pipeline completo: DAPT + NER (BERT e RoBERTa)
├── setup.bat                 # Instalação das dependências (Windows)
├── requirements.txt          # Dependências Python
├── .gitignore
└── README.md
```

> **Nota:** As pastas `tesemo_raw/`, `tesemo_clean/` e `experimentos/` não estão no repositório por serem muito grandes (>30 GB). Siga o guia abaixo para reproduzir os experimentos.

---

## Como Reproduzir

### 1. Pré-requisitos

- Python 3.10+
- CUDA 12.x (GPU NVIDIA recomendada — testado em RTX 5060 Ti 16GB)
- ~100 GB de espaço em disco
- Windows 10/11 ou Linux

### 2. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/JurisNER.git
cd JurisNER
```

### 3. Criar ambiente virtual e instalar dependências

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Baixar o corpus Ulysses Tesemô

O corpus está disponível no Google Drive (projeto Ulysses — Câmara dos Deputados):

- [Link 1 — Google Drive](https://drive.google.com/drive/folders/1hRugg8mC5R_COB11DI3O1qOBaaXJxdx0)
- [Link 2 — Google Drive](https://drive.google.com/drive/folders/1Sf9hNpoGO_hJtIvhsT1LvWyya0bnm70n)

Baixe e extraia o conteúdo para a pasta `tesemo_raw/`:

```
JurisNER/
└── tesemo_raw/
    ├── J1/   ← Documentos judiciais
    ├── L1/   ← Legislação federal
    ├── L2/   ← Legislação estadual
    └── ...
```

### 5. Limpar e dedupllicar o corpus

```bash
python tesemo_pipeline.py
```

Este script realiza, em paralelo:
- Filtro de tamanho mínimo (< 200 caracteres)
- Detecção de idioma (`langdetect`) — remove documentos fora do pt-BR
- Deduplicação exata via SHA-256
- Deduplicação fuzzy via MinHash LSH (threshold 85%)

Os arquivos limpos são salvos em `tesemo_clean/` e um relatório CSV é gerado.

> ⏱ Estimativa: ~1–2 horas (Ryzen 7 5700X, SSD)

### 6. Treinar e avaliar os modelos

**XLM-RoBERTa:**
```bash
# Certifique-se que TIPO_MODELO = "roberta" no script
python jurisroberta_pipeline.py
```

**BERTimbau:**
```bash
# Altere TIPO_MODELO = "bert" no script
python jurisroberta_pipeline.py
```

O pipeline executa automaticamente 4 etapas:

| Etapa | Descrição |
|---|---|
| **1 — Corpus** | Lê `tesemo_clean/` e gera `corpus.jsonl` |
| **2 — Tokenização** | Tokeniza e salva cache Arrow no SSD |
| **3 — DAPT/MLM** | Pré-treinamento adicional com Masked Language Modeling |
| **4 — NER** | Fine-tuning e avaliação no LENER-Br |

Os resultados são salvos em `experimentos/{bert,roberta}/ner/resultados.json`.

> ⏱ Estimativa total por modelo: ~12–20 horas (GPU RTX, corpus completo)

### 7. Inferência com modelo treinado

```bash
python jurisroberta_pipeline.py --inferir
```

Exemplo de saída:
```
======================================================================
TEXTO:
O réu João da Silva interpôs recurso ao STF com base no art. 5º da Constituição Federal.
======================================================================
[PESSOA            ] João da Silva (score=0.998)
[ORGANIZAÇÃO       ] STF (score=0.991)
[LEGISLACAO        ] art. 5º da Constituição Federal (score=0.987)
```

---

## Datasets Utilizados

| Dataset | Descrição | Link |
|---|---|---|
| **Ulysses Tesemô** | Corpus jurídico-legislativo brasileiro (~30 GB, 3,5M documentos) | [Paper](https://doi.org/10.1007/s10579-024-09762-8) |
| **LENER-Br** | Corpus anotado para NER jurídico em português (6 categorias de entidades) | [GitHub](https://github.com/peluz/lener-br) |

### Categorias de entidades do LENER-Br

| Entidade | Exemplo |
|---|---|
| `PESSOA` | João da Silva, Ministra Rosa Weber |
| `ORGANIZAÇÃO` | STF, Câmara dos Deputados, TJSP |
| `LOCAL` | São Paulo, Vara Federal de Brasília |
| `TEMPO` | 15 de março de 2024, prazo de 30 dias |
| `LEGISLAÇÃO` | art. 5º da CF, Lei nº 7.347/85 |
| `JURISPRUDÊNCIA` | RE 123.456/SP, Súmula 330 do STJ |

---

## Dependências Principais

```
torch>=2.0
transformers>=4.40
datasets>=2.18
seqeval
langdetect
datasketch
tqdm
```

Instale todas com:
```bash
pip install -r requirements.txt
```

---

## Trabalhos Relacionados

Este projeto se apoia nas seguintes referências principais:

- **BERTimbau** — Souza et al. (2020): modelo BERT para português brasileiro
- **LegalBERT** — Chalkidis et al. (2020): BERT especializado para textos jurídicos em inglês
- **JurisBERT** — Viegas et al. (2023): BERT jurídico para português brasileiro
- **RoBERTaLexPT** — Garcia et al. (2024): RoBERTa jurídico com deduplicação para português
- **Ulysses Tesemô** — Siqueira et al. (2024): corpus jurídico-legislativo em português

---

## Citação

Se você usar este projeto em sua pesquisa, por favor cite:

```bibtex
@monografia{costa2025jurisner,
  author    = {Bruno Martins Costa},
  title     = {Uma Adaptação da Arquitetura RoBERTa para o Domínio Jurídico Brasileiro},
  school    = {Universidade Federal de Catalão},
  year      = {2025},
  type      = {Trabalho de Conclusão de Curso},
  advisor   = {Marcio de Souza Dias}
}
```

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

Os dados do **Ulysses Tesemô** e do **LENER-Br** estão sujeitos às suas próprias licenças — consulte os repositórios originais antes de usar em produção.

---

## Contato

**Bruno Martins Costa**  
Universidade Federal de Catalão — UFCAT  
Curso de Bacharelado em Ciências da Computação
