# JurisNER — DAPT de BERTimbau e XLM-RoBERTa para NER jurídico em português

Código da monografia **"Adaptação de Domínio de Modelos Transformer para Reconhecimento de Entidades Nomeadas Jurídicas: uma Comparação entre BERTimbau e XLM-RoBERTa"** (Bruno Martins Costa, Ciência da Computação, UFCAT).

Compara `neuralmind/bert-base-portuguese-cased` (BERTimbau) e `xlm-roberta-base`, com e sem **Domain Adaptive Pretraining (DAPT)** sobre o corpus Ulysses-Tesemô, no NER do **LeNER-Br** (6 categorias: JURISPRUDÊNCIA, LEGISLAÇÃO, LOCAL, ORGANIZAÇÃO, PESSOA, TEMPO).

## Resultado principal

F1 micro no teste do LeNER-Br, média ± desvio-padrão de 5 sementes do ajuste fino (o DAPT foi executado **uma vez** por modelo):

| Modelo | Sem DAPT | Com DAPT | Δ (DAPT) | p (Welch) |
|---|---|---|---|---|
| BERTimbau | 89,08 ± 0,54 | 89,34 ± 0,73 | +0,25 p.p. | 0,55 |
| XLM-RoBERTa | 89,57 ± 0,69 | 90,31 ± 0,62 | +0,74 p.p. | 0,11 |

Entre modelos, com DAPT: −0,98 p.p. para o BERTimbau (p = 0,053); sem DAPT: −0,49 p.p. (p = 0,25).

**Leitura honesta:** o benefício do DAPT foi pequeno e não significativo com 5 sementes, e a vantagem do XLM-RoBERTa fica no limite da significância. Diferenças abaixo de ~1 p.p. não devem ser interpretadas como superioridade de um modelo. Tabelas completas (por entidade e por semente) em [`results/resumo_multiseed.md`](results/resumo_multiseed.md).

Limitações: uma execução de DAPT por modelo (1 época), 5 sementes, um único conjunto de dados.

## Estrutura

```
tesemo_pipeline.py          # limpeza/deduplicação do Tesemô (tamanho, idioma, SHA-256, MinHash LSH)
jurisroberta_pipeline.py    # corpus -> tokenização -> DAPT (MLM) -> NER (retomável)
ner_multiseed.py            # NER com várias sementes, com e sem DAPT; agrega e testa
requirements.txt            # versões exatas usadas
results/resumo_multiseed.md # resultados finais
legado/                     # pipeline antigo (ver nota abaixo)
```

`legado/jurisner_pipeline_antigo.py` é o script usado na primeira rodada do BERTimbau, que **não** usava o mesmo pipeline do XLM-RoBERTa; seus resultados não são comparáveis e não entram nas conclusões.

## Reproduzir

Ambiente dos experimentos: Windows 11, RTX 5060 Ti 16 GB (bf16), Ryzen 7 5700X, 16 GB RAM; `transformers 5.6.2`, `torch 2.11.0+cu128`.

```bash
python -m venv venv && venv\Scripts\activate      # Linux: source venv/bin/activate
python -m pip install -r requirements.txt
```

1. **Corpus.** Baixe o Ulysses-Tesemô (ver [artigo](https://doi.org/10.1007/s10579-024-09762-8)) em `tesemo_raw/` e rode `python tesemo_pipeline.py` (gera `tesemo_clean/`; 796.402 documentos, 13,17 GB após a curadoria).
2. **DAPT + NER (uma execução por modelo).** Em `jurisroberta_pipeline.py`, defina `TIPO_MODELO = "bert"` ou `"roberta"` e rode `python jurisroberta_pipeline.py`. Opções: `--so-ner` (pula DAPT, usa `experimentos/<modelo>/mlm`), `--inferir`. O DAPT salva checkpoints a cada 1000 passos e retoma sozinho se o processo for interrompido. Tempos observados: DAPT ≈ 6 h 25 (BERTimbau) e ≈ 10 h 16 (XLM-R); NER ≈ 26 min e ≈ 11 min.
3. **Sementes e baseline sem DAPT.**
   ```bash
   python ner_multiseed.py rodar     # padrão: bert+roberta, dapt+base, sementes 42–46
   python ner_multiseed.py agregar   # gera experimentos/multiseed/resumo_multiseed.md
   ```

Hiperparâmetros (iguais para os dois modelos): MLM 15 %, sequência 512, batch efetivo 32 (4×8), 1 época; NER 5 épocas, batch efetivo 32 (16×2), warmup 0,1, weight decay 0,01, AdamW fundido, bf16, melhor época por F1 de validação.

Nota de reprodutibilidade: o XLM-R foi adaptado com a versão do pipeline anterior à adição da retomada de checkpoint (diferença apenas na política de salvamento, sem efeito nos hiperparâmetros). Mesmo com a mesma semente, o F1 varia entre execuções por não determinismo da GPU, por isso os resultados são reportados como média de sementes.

## Dados

- **Ulysses-Tesemô** — corpus jurídico-legislativo brasileiro.
- **LeNER-Br** — [github.com/peluz/lener-br](https://github.com/peluz/lener-br) (baixado automaticamente pelo pipeline).

Os dados estão sujeitos às licenças originais. **Licença do código:** a definir pelo autor.

## Citação

```bibtex
@monografia{costa2026jurisner,
  author = {Bruno Martins Costa},
  title  = {Adapta{\c c}{\~a}o de Dom{\'\i}nio de Modelos Transformer para Reconhecimento de Entidades Nomeadas Jur{\'\i}dicas: uma Compara{\c c}{\~a}o entre BERTimbau e XLM-RoBERTa},
  school = {Universidade Federal de Catal{\~a}o},
  year   = {2026},
  type   = {Trabalho de Conclus{\~a}o de Curso}
}
```
