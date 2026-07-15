"""
JurisNER — Comparação entre BERT e RoBERTa para NER Jurídico Brasileiro
========================================================================
Autor: Bruno Martins Costa (TCC - UFCAT 2025)
Orientador: Prof. Dr. Marcio de Souza Dias

OTIMIZAÇÕES DESTA VERSÃO:
  CORPUS / I-O
    - Sem limite de GB — usa todo o tesemo_clean/
    - Etapa 1 escreve corpus.jsonl em streaming (sem carregar tudo na RAM)
    - Etapa 2 pré-tokeniza e salva em formato Arrow (HuggingFace cache no SSD)
      → nas próximas execuções a tokenização é pulada completamente

  MLM (DAPT)
    - bf16 nativo (Blackwell)
    - gradient_checkpointing
    - dataloader_num_workers=4  (CPU prepara batches enquanto GPU treina)
    - dataloader_pin_memory=True
    - torch_compile=False (Windows nao suporta Triton)
    - optim="adamw_torch_fused"  → AdamW fundido, ~10% mais rápido que padrão
    - save_total_limit=1         → só mantém 1 checkpoint no disco

  NER
    - Todas as otimizações acima
    - padding dinâmico + pad_to_multiple_of=8
    - num_proc=4 na tokenização
    - batch_size 16, grad_acc 2
    - Dataset pré-tokenizado salvo em Arrow (cache SSD)

COMO USAR:
  python jurisner_pipeline.py
  python jurisner_pipeline.py --so-ner    # pula etapas 1-3, só roda NER
  python jurisner_pipeline.py --inferir   # só roda inferência
"""

import os
import sys
import json
import hashlib
import logging
import re

from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

SEED = 42

# "bert"    -> BERTimbau
# "roberta" -> XLM-RoBERTa
TIPO_MODELO = "bert"

MODELOS = {
    "bert": {
        "nome": "BERTimbau",
        "hf":   "neuralmind/bert-base-portuguese-cased",
    },
    "roberta": {
        "nome": "XLM-RoBERTa",
        "hf":   "xlm-roberta-base",
    },
}

MODELO_BASE = MODELOS[TIPO_MODELO]["hf"]
NOME_MODELO = MODELOS[TIPO_MODELO]["nome"]

# =============================================================================
# HIPERPARÂMETROS
# =============================================================================

MAX_SEQ_LENGTH = 512
MLM_PROB       = 0.15

# ── MLM ──────────────────────────────────────────────────────────────────────
BATCH_MLM    = 4     # dobrado em relação ao anterior (bf16 + grad_ckpt dão folga)
GRAD_ACC_MLM = 8     # batch efetivo = 4 * 8 = 32
EPOCHS_MLM   = 1

# ── NER ──────────────────────────────────────────────────────────────────────
BATCH_NER    = 16
GRAD_ACC_NER = 2     # batch efetivo = 16 * 2 = 32
EPOCHS_NER   = 5

# CPU workers — Ryzen 7 5700X tem folga enquanto GPU treina
DATALOADER_WORKERS = 4

# =============================================================================
# DIRETÓRIOS
# =============================================================================

BASE_DIR    = Path(f"experimentos/{TIPO_MODELO}")
DIR_CORPUS  = BASE_DIR / "corpus"
DIR_CACHE   = BASE_DIR / "cache"    # datasets Arrow pré-tokenizados
DIR_MLM     = BASE_DIR / "mlm"
DIR_NER     = BASE_DIR / "ner"
DIR_LOGS    = BASE_DIR / "logs"

for d in [DIR_CORPUS, DIR_CACHE, DIR_MLM, DIR_NER, DIR_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DIR_LOGS / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# ETAPA 1 — CORPUS LOCAL (streaming, sem limite de GB)
# =============================================================================

def etapa1_preparar_corpus(tesemo_clean_dir: str = "./tesemo_clean"):

    log.info("=" * 70)
    log.info("ETAPA 1 — Preparando corpus local (tesemo_clean/) — SEM LIMITE")
    log.info("=" * 70)

    arquivo_saida = DIR_CORPUS / "corpus.jsonl"

    if arquivo_saida.exists():
        tamanho = arquivo_saida.stat().st_size / 1e9
        log.info(f"corpus.jsonl já existe ({tamanho:.2f} GB). Pulando.")
        return str(arquivo_saida)

    pasta = Path(tesemo_clean_dir)
    if not pasta.exists():
        raise FileNotFoundError(
            f"Pasta '{tesemo_clean_dir}' não encontrada.\n"
            "Execute primeiro o tesemo_pipeline.py para gerar o corpus limpo."
        )

    arquivos    = sorted(pasta.rglob("*.txt"))
    bytes_total = 0
    docs_ok     = 0
    docs_curtos = 0

    log.info(f"Arquivos encontrados: {len(arquivos):,}")

    # Escrita em streaming — nunca carrega tudo na RAM
    with open(arquivo_saida, "w", encoding="utf-8") as f_out:
        for caminho in tqdm(arquivos, desc="Lendo corpus", unit="arq"):
            try:
                texto = caminho.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                try:
                    texto = caminho.read_text(encoding="latin-1").strip()
                except Exception:
                    continue

            if len(texto) < 100:
                docs_curtos += 1
                continue

            linha = json.dumps({"text": texto}, ensure_ascii=False)
            f_out.write(linha + "\n")
            bytes_total += len(linha.encode("utf-8"))
            docs_ok     += 1

    log.info(f"Documentos incluídos : {docs_ok:,}")
    log.info(f"Documentos curtos    : {docs_curtos:,}")
    log.info(f"Volume total         : {bytes_total/1e9:.2f} GB")
    log.info(f"Corpus salvo em      : {arquivo_saida}")

    return str(arquivo_saida)

# =============================================================================
# ETAPA 2 — PRÉ-PROCESSAMENTO + CACHE ARROW
# =============================================================================

def etapa2_preprocessar(caminho_corpus):
    """
    O corpus já foi limpo e deduplicado pelo tesemo_pipeline.py.
    Aqui só fazemos a tokenização e salvamos em formato Arrow no SSD.
    Na próxima execução o cache é carregado direto — tokenização pulada.
    """

    log.info("=" * 70)
    log.info("ETAPA 2 — Tokenização + cache Arrow no SSD")
    log.info("=" * 70)

    cache_mlm_dir = DIR_CACHE / "mlm_tokenizado"

    # Cache já existe → pula tudo
    if cache_mlm_dir.exists() and any(cache_mlm_dir.iterdir()):
        log.info("Cache Arrow do MLM já existe. Pulando etapa 2.")
        return cache_mlm_dir

    log.info("Tokenizando corpus.jsonl e salvando cache Arrow no SSD...")
    log.info("(Isso pode demorar bastante — só acontece uma vez)")

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    # Carrega o corpus.jsonl em streaming — não carrega tudo na RAM
    dataset = load_dataset(
        "json",
        data_files={"train": caminho_corpus},
        split="train",
        cache_dir=str(DIR_CACHE / "raw"),
    )

    split = dataset.train_test_split(test_size=0.1, seed=SEED)

    def tokenizar(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    ds_tok = split.map(
        tokenizar,
        batched=True,
        batch_size=1000,
        num_proc=4,               # 4 núcleos do Ryzen em paralelo
        remove_columns=["text"],
        desc="Tokenizando",
    )

    ds_tok.save_to_disk(str(cache_mlm_dir))
    log.info(f"Cache Arrow salvo em: {cache_mlm_dir}")

    return cache_mlm_dir

# =============================================================================
# ETAPA 3 — DAPT / MLM  (totalmente otimizado)
# =============================================================================

def etapa3_mlm(cache_mlm_dir: str):

    log.info("=" * 70)
    log.info("ETAPA 3 — Domain Adaptive Pretraining (MLM)")
    log.info("=" * 70)

    if (DIR_MLM / "config.json").exists():
        log.info("Modelo MLM já existe. Pulando.")
        return str(DIR_MLM)

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from datasets import DatasetDict, load_from_disk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    model     = AutoModelForMaskedLM.from_pretrained(MODELO_BASE)

    # Gradient checkpointing — troca um pouco de velocidade por VRAM
    # permite batch maior sem OOM
    model.gradient_checkpointing_enable()
    model.to(device)

    # Carrega dataset pré-tokenizado do cache Arrow (muito mais rápido)
    log.info("Carregando dataset do cache Arrow...")
    ds_tok = load_from_disk(cache_mlm_dir)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
        pad_to_multiple_of=8,    # melhor para tensor cores RTX
    )

    args = TrainingArguments(
        output_dir=str(DIR_MLM),

        num_train_epochs=EPOCHS_MLM,

        per_device_train_batch_size=BATCH_MLM,
        per_device_eval_batch_size=BATCH_MLM,
        gradient_accumulation_steps=GRAD_ACC_MLM,

        # bf16 nativo no Blackwell — mais rápido E mais estável que fp16
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,          # mantém só 1 checkpoint → economiza SSD

        load_best_model_at_end=True,
        logging_steps=200,
        warmup_ratio=0.05,
        weight_decay=0.01,

        # AdamW fundido — implementação CUDA do otimizador, ~10% mais rápido
        optim="adamw_torch_fused",


        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,  # transferência CPU→GPU mais rápida

        # demora ~3-5 min na primeira vez, vale a pena para treinos longos
        torch_compile=False,  # Triton não suportado no Windows

        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    log.info("Iniciando DAPT/MLM...")
    trainer.train()

    model.save_pretrained(str(DIR_MLM))
    tokenizer.save_pretrained(str(DIR_MLM))
    log.info("MLM concluído!")

    return str(DIR_MLM)

# =============================================================================
# ETAPA 4 — NER  (totalmente otimizado)
# =============================================================================

def etapa4_ner(caminho_modelo_mlm):

    log.info("=" * 70)
    log.info("ETAPA 4 — Fine-tuning NER")
    log.info("=" * 70)

    if (DIR_NER / "config.json").exists():
        log.info("Modelo NER já existe. Pulando.")
        return str(DIR_NER)

    import torch
    import numpy as np
    import urllib.request

    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset, DatasetDict, load_from_disk
    from seqeval.metrics import classification_report, f1_score

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Download LENER-Br ─────────────────────────────────────────────────────
    dir_lener = DIR_CORPUS / "lener_br"
    dir_lener.mkdir(exist_ok=True)
    BASE_URL  = "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br"
    splits_urls = {
        "train":      f"{BASE_URL}/train/train.conll",
        "validation": f"{BASE_URL}/dev/dev.conll",
        "test":       f"{BASE_URL}/test/test.conll",
    }
    for split, url in splits_urls.items():
        destino = dir_lener / f"{split}.conll"
        if not destino.exists():
            log.info(f"Baixando LENER-Br {split}...")
            urllib.request.urlretrieve(url, destino)

    # ── Leitura CoNLL ─────────────────────────────────────────────────────────
    def ler_conll(caminho):
        sentences_tokens, sentences_tags = [], []
        tokens, tags = [], []
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if linha == "":
                    if tokens:
                        sentences_tokens.append(tokens)
                        sentences_tags.append(tags)
                        tokens, tags = [], []
                else:
                    partes = linha.split()
                    tokens.append(partes[0])
                    tags.append(partes[-1])
        if tokens:
            sentences_tokens.append(tokens)
            sentences_tags.append(tags)
        return sentences_tokens, sentences_tags

    # ── Cache Arrow do NER ────────────────────────────────────────────────────
    cache_ner_dir = DIR_CACHE / "ner_tokenizado"

    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo_mlm)

    if cache_ner_dir.exists() and any(cache_ner_dir.iterdir()):
        log.info("Cache Arrow do NER encontrado. Carregando...")
        lener_tok = load_from_disk(str(cache_ner_dir))

        # Reconstrói labels a partir do cache
        todas_tags = set()
        for split in ["train", "validation", "test"]:
            toks, tgs = ler_conll(dir_lener / f"{split}.conll")
            for tags in tgs:
                todas_tags.update(tags)
        LABELS   = ["O"] + sorted(t for t in todas_tags if t != "O")
        label2id = {l: i for i, l in enumerate(LABELS)}
        id2label = {i: l for l, i in label2id.items()}
    else:
        # Monta dataset do zero e salva cache Arrow
        splits_data = {}
        todas_tags  = set()

        for split in ["train", "validation", "test"]:
            toks, tgs = ler_conll(dir_lener / f"{split}.conll")
            splits_data[split] = Dataset.from_dict({
                "tokens":   toks,
                "ner_tags": tgs,
            })
            for tags in tgs:
                todas_tags.update(tags)

        lener    = DatasetDict(splits_data)
        LABELS   = ["O"] + sorted(t for t in todas_tags if t != "O")
        label2id = {l: i for i, l in enumerate(LABELS)}
        id2label = {i: l for l, i in label2id.items()}

        log.info(f"Labels NER: {LABELS}")

        def tokenizar_alinhar(exemplos):
            enc = tokenizer(
                exemplos["tokens"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                is_split_into_words=True,
                padding=False,
            )
            labels_batch = []
            for i, tags in enumerate(exemplos["ner_tags"]):
                word_ids = enc.word_ids(batch_index=i)
                prev, ids = None, []
                for wid in word_ids:
                    if wid is None:
                        ids.append(-100)
                    elif wid != prev:
                        ids.append(label2id.get(tags[wid], 0) if wid < len(tags) else -100)
                    else:
                        ids.append(-100)
                    prev = wid
                labels_batch.append(ids)
            enc["labels"] = labels_batch
            return enc

        log.info("Tokenizando NER (com cache Arrow)...")
        lener_tok = lener.map(
            tokenizar_alinhar,
            batched=True,
            num_proc=4,
            remove_columns=lener["train"].column_names,
            desc="Tokenizando NER",
        )

        lener_tok.save_to_disk(str(cache_ner_dir))
        log.info(f"Cache NER salvo em: {cache_ner_dir}")

    log.info(f"Labels NER: {LABELS}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = AutoModelForTokenClassification.from_pretrained(
        caminho_modelo_mlm,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.gradient_checkpointing_enable()
    model.to(device)


    # ── Collator com padding dinâmico ─────────────────────────────────────────
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ── Métricas ──────────────────────────────────────────────────────────────
    def metricas(pred):
        logits, labels = pred
        preds  = np.argmax(logits, axis=-1)
        pv = [[id2label[p] for p, l in zip(pr, la) if l != -100]
              for pr, la in zip(preds, labels)]
        lv = [[id2label[l] for p, l in zip(pr, la) if l != -100]
              for pr, la in zip(preds, labels)]
        rel        = classification_report(lv, pv, output_dict=True, zero_division=0)
        resultados = {"f1": f1_score(lv, pv)}
        for entidade in rel:
            if isinstance(rel[entidade], dict) and "f1-score" in rel[entidade]:
                resultados[f"f1_{entidade.lower()}"] = rel[entidade]["f1-score"]
        return resultados

    # ── TrainingArguments ─────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(DIR_NER),

        num_train_epochs=EPOCHS_NER,

        per_device_train_batch_size=BATCH_NER,
        per_device_eval_batch_size=BATCH_NER,
        gradient_accumulation_steps=GRAD_ACC_NER,

        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="f1",

        logging_steps=50,
        warmup_ratio=0.1,
        weight_decay=0.01,

        optim="adamw_torch_fused",


        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,

        torch_compile=False,  # Triton não suportado no Windows

        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lener_tok["train"],
        eval_dataset=lener_tok["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=metricas,
    )

    log.info("Treinando NER...")
    trainer.train()

    trainer.save_model(str(DIR_NER))
    tokenizer.save_pretrained(str(DIR_NER))

    # ── Teste final ───────────────────────────────────────────────────────────
    log.info("Avaliando no conjunto de TESTE...")
    res = trainer.evaluate(eval_dataset=lener_tok["test"])

    log.info("=" * 70)
    log.info("RESULTADOS NO TESTE")
    log.info("=" * 70)
    for k, v in res.items():
        if isinstance(v, float):
            log.info(f"  {k:35s}: {v:.4f}")

    with open(DIR_NER / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    return str(DIR_NER)

# =============================================================================
# INFERÊNCIA
# =============================================================================

def inferencia(texto, caminho_modelo=None):
    from transformers import pipeline

    ner = pipeline(
        "ner",
        model=caminho_modelo or str(DIR_NER),
        tokenizer=caminho_modelo or str(DIR_NER),
        aggregation_strategy="simple",
        device=0,
    )

    resultados = ner(texto)
    print("\n" + "=" * 70)
    print("TEXTO:")
    print(texto)
    print("=" * 70)
    for e in resultados:
        print(f"[{e['entity_group']:18s}] {e['word']} (score={e['score']:.3f})")
    return resultados

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    log.info("=" * 70)
    log.info(f"MODELO     : {NOME_MODELO}")
    log.info(f"HuggingFace: {MODELO_BASE}")
    log.info("=" * 70)

    import torch
    if torch.cuda.is_available():
        log.info(f"GPU : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        log.info(f"bf16: {torch.cuda.is_bf16_supported()}")
    else:
        log.warning("CUDA não detectado!")

    so_ner  = "--so-ner"  in sys.argv
    inferir = "--inferir" in sys.argv

    if inferir:
        exemplos = [
            "O réu João da Silva interpôs recurso ao STF com base no art. 5º da Constituição Federal.",
            "A Câmara dos Deputados aprovou em 15 de março de 2024 o Projeto de Lei nº 1.234/2023.",
            "O Ministério Público do Estado de São Paulo ingressou com ação civil pública conforme a Lei nº 7.347/85.",
        ]
        for texto in exemplos:
            inferencia(texto)

    elif so_ner:
        log.info("Modo --so-ner: pulando etapas 1-3.")
        etapa4_ner(str(DIR_MLM))

    else:
        corpus_raw      = etapa1_preparar_corpus("./tesemo_clean")
        cache_mlm       = etapa2_preprocessar(corpus_raw)
        modelo_mlm      = etapa3_mlm(str(cache_mlm))
        modelo_ner      = etapa4_ner(modelo_mlm)

        exemplos = [
            "O réu João da Silva interpôs recurso ao STF com base no art. 5º da Constituição Federal.",
            "A Câmara dos Deputados aprovou em 15 de março de 2024 o Projeto de Lei nº 1.234/2023.",
            "O Ministério Público do Estado de São Paulo ingressou com ação civil pública conforme a Lei nº 7.347/85.",
        ]
        for texto in exemplos:
            inferencia(texto, modelo_ner)

    log.info("=" * 70)
    log.info("CONCLUÍDO!")
    log.info("=" * 70)
