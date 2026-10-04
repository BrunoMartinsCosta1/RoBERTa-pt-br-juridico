# Resultados NER (LeNER-Br, teste) — 5 sementes (42–46), média ± desvio-padrão (ddof=1), F1 em %

Saída de `python ner_multiseed.py agregar`.

| modelo | config | n | f1 | macro avg | jurisprudencia | legislacao | local | organizacao | pessoa | tempo | melhor_val_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bert | base | 5 | 89.08 ± 0.54 | 86.56 ± 0.55 | 80.96 ± 1.01 | 94.19 ± 0.74 | 69.06 ± 2.87 | 86.08 ± 1.21 | 94.10 ± 0.78 | 94.94 ± 0.72 | 86.91 ± 0.54 |
| bert | dapt | 5 | 89.34 ± 0.73 | 86.76 ± 1.21 | 82.54 ± 2.84 | 94.82 ± 0.54 | 68.62 ± 4.34 | 85.88 ± 0.59 | 94.78 ± 1.05 | 93.91 ± 1.62 | 87.95 ± 0.80 |
| roberta | base | 5 | 89.57 ± 0.69 | 87.17 ± 0.45 | 85.40 ± 1.38 | 94.13 ± 0.83 | 68.06 ± 2.63 | 85.48 ± 1.51 | 95.48 ± 0.59 | 94.50 ± 1.22 | 89.05 ± 0.73 |
| roberta | dapt | 5 | 90.31 ± 0.62 | 87.99 ± 1.21 | 85.86 ± 2.39 | 95.28 ± 0.76 | 70.39 ± 4.85 | 86.43 ± 0.90 | 95.56 ± 0.38 | 94.43 ± 1.12 | 89.69 ± 0.55 |

"base" = checkpoint original do Hugging Face (sem DAPT); "dapt" = após DAPT (1 época, 1 execução por modelo).

## F1 de teste por semente
| modelo | config | semente | F1 teste | época do melhor val |
|---|---|---|---|---|
| bert | base | 42 | 90.00 | 4 |
| bert | base | 43 | 88.73 | 4 |
| bert | base | 44 | 89.11 | 3 |
| bert | base | 45 | 88.92 | 3 |
| bert | base | 46 | 88.67 | 4 |
| bert | dapt | 42 | 88.92 | 3 |
| bert | dapt | 43 | 89.86 | 5 |
| bert | dapt | 44 | 88.92 | 5 |
| bert | dapt | 45 | 90.34 | 5 |
| bert | dapt | 46 | 88.64 | 4 |
| roberta | base | 42 | 89.35 | 5 |
| roberta | base | 43 | 89.97 | 5 |
| roberta | base | 44 | 90.23 | 5 |
| roberta | base | 45 | 89.82 | 5 |
| roberta | base | 46 | 88.49 | 5 |
| roberta | dapt | 42 | 91.03 | 5 |
| roberta | dapt | 43 | 90.67 | 5 |
| roberta | dapt | 44 | 89.46 | 5 |
| roberta | dapt | 45 | 90.50 | 4 |
| roberta | dapt | 46 | 89.92 | 5 |

## Comparações (diferença de médias do F1 de teste)
| comparação | Δ | p (Welch) | p (Mann-Whitney) |
|---|---|---|---|
| bert: DAPT − sem DAPT | +0.25 p.p. | 0.5537 | 0.8413 |
| roberta: DAPT − sem DAPT | +0.74 p.p. | 0.1112 | 0.1508 |
| DAPT: BERTimbau − XLM-R | -0.98 p.p. | 0.0527 | 0.0556 |
| sem DAPT: BERTimbau − XLM-R | -0.49 p.p. | 0.2497 | 0.4206 |

Com n=5 por grupo o poder estatístico é baixo (o menor p possível no Mann-Whitney é 0,0079).
