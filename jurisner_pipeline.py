"""
JurisNER — Comparação entre BERT e RoBERTa para NER Jurídico Brasileiro
========================================================================

OBJETIVO DO TCC
---------------
Comparar o desempenho de duas arquiteturas Transformer em tarefas de
Reconhecimento de Entidades Nomeadas (NER) no domínio jurídico brasileiro:

    1. BERTimbau
    2. XLM-RoBERTa

PIPELINE:
    1. Download do corpus jurídico
    2. Pré-processamento
    3. Domain Adaptive Pretraining (DAPT / MLM)
    4. Fine-tuning NER (LENER-Br)
    5. Avaliação
    6. Inferência

OBS:
    - NÃO usa tokenizador customizado
    - Mantém tokenizer original de cada arquitetura
    - Experimento cientificamente mais controlado

COMO USAR:
    python jurisner_pipeline.py
"""

import os
import sys
import json
import hashlib
import logging
import re

print("ARQUIVO NOVO CARREGADO")

from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

SEED = 42

# -------------------------
# ESCOLHA DO MODELO
# -------------------------
#
# Opções:
#
#   "bert"     -> BERTimbau
#   "roberta"  -> XLM-RoBERTa
#
# Troque aqui para comparar os modelos
#

TIPO_MODELO = "bert"

# =============================================================================
# MODELOS
# =============================================================================

MODELOS = {
    "bert": {
        "nome": "BERTimbau",
        "hf": "neuralmind/bert-base-portuguese-cased",
    },

    "roberta": {
        "nome": "XLM-RoBERTa",
        "hf": "xlm-roberta-base",
    }
}

MODELO_BASE = MODELOS[TIPO_MODELO]["hf"]
NOME_MODELO = MODELOS[TIPO_MODELO]["nome"]

# =============================================================================
# HIPERPARÂMETROS
# =============================================================================

TAMANHO_MAXIMO_CORPUS_GB = 5

MAX_SEQ_LENGTH = 512

MLM_PROB = 0.15

# MAIS SEGURO PARA 16GB VRAM
BATCH_TRAIN = 2
BATCH_NER = 8

# Acumulação para simular batch maior
GRAD_ACC_MLM = 16
GRAD_ACC_NER = 2

EPOCHS_MLM = 1
EPOCHS_NER = 5

# =============================================================================
# DIRETÓRIOS
# =============================================================================

BASE_DIR = Path(f"experimentos/{TIPO_MODELO}")

DIR_CORPUS = BASE_DIR / "corpus"
DIR_MLM = BASE_DIR / "mlm"
DIR_NER = BASE_DIR / "ner"
DIR_LOGS = BASE_DIR / "logs"

for d in [DIR_CORPUS, DIR_MLM, DIR_NER, DIR_LOGS]:
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
# ETAPA 1 — DOWNLOAD DO CORPUS
# =============================================================================

def etapa1_baixar_corpus():

    log.info("=" * 70)
    log.info("ETAPA 1 — Download do corpus jurídico")
    log.info("=" * 70)

    arquivo_saida = DIR_CORPUS / "corpus.jsonl"

    if arquivo_saida.exists():
        log.info("Corpus já existe. Pulando.")
        return str(arquivo_saida)

    from datasets import load_dataset

    dataset = load_dataset(
        "eduagarcia/LegalPT_dedup",
        name="tesemo_v2",
        split="train",
        streaming=True,
    )

    limite_bytes = TAMANHO_MAXIMO_CORPUS_GB * 1_000_000_000

    bytes_total = 0
    docs = 0

    with open(arquivo_saida, "w", encoding="utf-8") as f:

        barra = tqdm(desc="Baixando", unit=" docs")

        for ex in dataset:

            texto = ex.get("text", "").strip()

            if len(texto) < 100:
                continue

            linha = json.dumps(
                {"text": texto},
                ensure_ascii=False
            )

            f.write(linha + "\n")

            bytes_total += len(linha.encode("utf-8"))
            docs += 1

            barra.update(1)

            barra.set_postfix({
                "docs": docs,
                "GB": f"{bytes_total/1e9:.2f}"
            })

            if bytes_total >= limite_bytes:
                break

        barra.close()

    log.info(f"Corpus salvo: {arquivo_saida}")

    return str(arquivo_saida)

# =============================================================================
# ETAPA 2 — PRÉ-PROCESSAMENTO
# =============================================================================

def etapa2_preprocessar(caminho_corpus):

    log.info("=" * 70)
    log.info("ETAPA 2 — Pré-processamento")
    log.info("=" * 70)

    arquivo_saida = DIR_CORPUS / "corpus_limpo.txt"

    if arquivo_saida.exists():
        log.info("Corpus limpo já existe. Pulando.")
        return str(arquivo_saida)

    hashes = set()

    total = 0
    duplicados = 0
    blocos = 0

    with (
        open(caminho_corpus, "r", encoding="utf-8") as entrada,
        open(arquivo_saida, "w", encoding="utf-8") as saida,
    ):

        for linha in tqdm(entrada, desc="Limpando"):

            total += 1

            try:
                texto = json.loads(linha)["text"]
            except:
                continue

            # Normaliza espaços
            texto = re.sub(r"\s+", " ", texto)

            # Remove caracteres estranhos
            texto = re.sub(
                r"[^\w\s.,;:!?()\-\u2013\u2014\"'º°§]",
                "",
                texto
            )

            texto = texto.strip()

            if len(texto) < 50:
                continue

            # Deduplicação
            h = hashlib.md5(texto.encode()).hexdigest()

            if h in hashes:
                duplicados += 1
                continue

            hashes.add(h)

            # Segmentação
            palavras = texto.split()

            for i in range(0, len(palavras), 400):

                bloco = " ".join(palavras[i:i+400])

                if len(bloco) > 50:
                    saida.write(bloco + "\n")
                    blocos += 1

    log.info(f"Total       : {total:,}")
    log.info(f"Duplicados  : {duplicados:,}")
    log.info(f"Blocos      : {blocos:,}")

    return str(arquivo_saida)

# =============================================================================
# ETAPA 3 — DAPT / MLM
# =============================================================================

def etapa3_mlm(caminho_corpus_limpo):

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

    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Dispositivo: {device}")

    if device == "cuda":
        log.info(f"GPU : {torch.cuda.get_device_name(0)}")

    # Tokenizer ORIGINAL
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    # Modelo ORIGINAL
    model = AutoModelForMaskedLM.from_pretrained(MODELO_BASE)

    model.to(device)

    # Dataset lazy
    dataset = load_dataset(
        "text",
        data_files={"train": caminho_corpus_limpo},
        split="train",
    )

    split = dataset.train_test_split(
        test_size=0.1,
        seed=SEED
    )

    ds_train = split["train"]
    ds_valid = split["test"]

    log.info(f"Treino    : {len(ds_train):,}")
    log.info(f"Validação : {len(ds_valid):,}")

    def tokenizar(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    log.info("Tokenizando treino...")
    ds_train = ds_train.map(
        tokenizar,
        batched=True,
        remove_columns=["text"],
    )

    log.info("Tokenizando validação...")
    ds_valid = ds_valid.map(
        tokenizar,
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    args = TrainingArguments(
        output_dir=str(DIR_MLM),

        num_train_epochs=EPOCHS_MLM,

        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_TRAIN,

        gradient_accumulation_steps=GRAD_ACC_MLM,

        # Melhor para Blackwell
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        logging_steps=100,

        warmup_steps=500,

        weight_decay=0.01,

        report_to="none",

        dataloader_num_workers=0,

        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,

        train_dataset=ds_train,
        eval_dataset=ds_valid,

        data_collator=collator,

        processing_class=tokenizer,
    )

    log.info("Treinando MLM...")
    trainer.train()

    model.save_pretrained(str(DIR_MLM))
    tokenizer.save_pretrained(str(DIR_MLM))

    log.info("MLM concluído!")

    return str(DIR_MLM)

# =============================================================================
# ETAPA 4 — NER
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

    from datasets import Dataset, DatasetDict

    from seqeval.metrics import (
        classification_report,
        f1_score,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # DOWNLOAD LENER-BR (GitHub oficial)
    # =========================================================================

    log.info("Baixando LENER-Br do GitHub...")

    dir_lener = DIR_CORPUS / "lener_br"
    dir_lener.mkdir(exist_ok=True)

    BASE_URL = "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br"

    arquivos = {
        "train":      f"{BASE_URL}/train/train.conll",
        "validation": f"{BASE_URL}/dev/dev.conll",
        "test":       f"{BASE_URL}/test/test.conll",
    }

    for split, url in arquivos.items():
        destino = dir_lener / f"{split}.conll"
        if not destino.exists():
            log.info(f"Baixando {split}...")
            urllib.request.urlretrieve(url, destino)

    # =========================================================================
    # LEITURA DOS ARQUIVOS CoNLL
    # =========================================================================

    def ler_conll(caminho):
        sentences_tokens = []
        sentences_tags   = []
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

    splits_data = {}
    for split in ["train", "validation", "test"]:
        toks, tgs = ler_conll(dir_lener / f"{split}.conll")
        splits_data[split] = Dataset.from_dict({
            "tokens":   toks,
            "ner_tags": tgs,
        })

    lener = DatasetDict(splits_data)

    log.info(lener)

    # =========================================================================
    # LABELS
    # =========================================================================

    todas_tags = set()
    for split in ["train", "validation", "test"]:
        for tags in lener[split]["ner_tags"]:
            todas_tags.update(tags)

    LABELS = ["O"] + sorted(t for t in todas_tags if t != "O")

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    log.info(f"Labels encontradas: {LABELS}")

    # =========================================================================
    # TOKENIZER
    # =========================================================================

    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo_mlm)

    # =========================================================================
    # MODELO
    # =========================================================================

    model = AutoModelForTokenClassification.from_pretrained(
        caminho_modelo_mlm,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.gradient_checkpointing_enable()
    model.to(device)

    # =========================================================================
    # TOKENIZAÇÃO + ALINHAMENTO
    # =========================================================================

    def tokenizar_alinhar(exemplos):

        enc = tokenizer(
            exemplos["tokens"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            is_split_into_words=True,
        )

        labels = []

        for i, tags in enumerate(exemplos["ner_tags"]):

            word_ids = enc.word_ids(batch_index=i)
            prev = None
            ids = []

            for wid in word_ids:
                if wid is None:
                    ids.append(-100)
                elif wid != prev:
                    if wid < len(tags):
                        ids.append(label2id.get(tags[wid], 0))
                    else:
                        ids.append(-100)
                else:
                    ids.append(-100)
                prev = wid

            labels.append(ids)

        enc["labels"] = labels
        return enc

    cols = lener["train"].column_names

    log.info("Tokenizando dataset NER...")

    lener_tok = lener.map(
        tokenizar_alinhar,
        batched=True,
        remove_columns=cols,
    )

    # =========================================================================
    # COLLATOR
    # =========================================================================

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # =========================================================================
    # MÉTRICAS
    # =========================================================================

    def metricas(pred):

        logits, labels = pred
        preds = np.argmax(logits, axis=-1)

        pv = [
            [id2label[p] for p, l in zip(pr, la) if l != -100]
            for pr, la in zip(preds, labels)
        ]

        lv = [
            [id2label[l] for p, l in zip(pr, la) if l != -100]
            for pr, la in zip(preds, labels)
        ]

        rel = classification_report(lv, pv, output_dict=True, zero_division=0)

        resultados = {"f1": f1_score(lv, pv)}

        for entidade in rel:
            if isinstance(rel[entidade], dict) and "f1-score" in rel[entidade]:
                resultados[f"f1_{entidade.lower()}"] = rel[entidade]["f1-score"]

        return resultados

    # =========================================================================
    # TRAINING ARGUMENTS
    # =========================================================================

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
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        warmup_steps=200,
        weight_decay=0.01,
        report_to="none",
        dataloader_num_workers=0,
        seed=SEED,
    )

    # =========================================================================
    # TRAINER
    # =========================================================================

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lener_tok["train"],
        eval_dataset=lener_tok["validation"],
        processing_class=tokenizer,   # <- corrigido
        data_collator=collator,
        compute_metrics=metricas,
    )

    # =========================================================================
    # TREINAMENTO
    # =========================================================================

    log.info("Treinando NER...")
    trainer.train()

    # =========================================================================
    # SALVA MODELO
    # =========================================================================

    trainer.save_model(str(DIR_NER))
    tokenizer.save_pretrained(str(DIR_NER))

    # =========================================================================
    # TESTE FINAL
    # =========================================================================

    log.info("Avaliando TESTE...")
    res = trainer.evaluate(eval_dataset=lener_tok["test"])

    log.info("=" * 70)
    log.info("RESULTADOS")
    log.info("=" * 70)

    for k, v in res.items():
        if isinstance(v, float):
            log.info(f"{k:35s}: {v:.4f}")

    with open(DIR_NER / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    log.info(f"Modelo salvo em: {DIR_NER}")

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

        print(
            f"[{e['entity_group']:18s}] "
            f"{e['word']} "
            f"(score={e['score']:.3f})"
        )

    return resultados

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    log.info("=" * 70)
    log.info(f"MODELO: {NOME_MODELO}")
    log.info(f"HuggingFace: {MODELO_BASE}")
    log.info("=" * 70)

    import torch

    if torch.cuda.is_available():

        log.info(f"GPU : {torch.cuda.get_device_name(0)}")

        log.info(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB"
        )

    else:
        log.warning("CUDA não detectado!")

    # ETAPAS
    corpus_raw = etapa1_baixar_corpus()

    corpus_limpo = etapa2_preprocessar(corpus_raw)

    modelo_mlm = etapa3_mlm(corpus_limpo)

    modelo_ner = etapa4_ner(modelo_mlm)

    # TESTES
    exemplos = [

        "O réu João da Silva interpôs recurso ao STF com base no art. 5º da Constituição Federal.",

        "A Câmara dos Deputados aprovou em 15 de março de 2024 o Projeto de Lei nº 1.234/2023.",

        "O Ministério Público do Estado de São Paulo ingressou com ação civil pública conforme a Lei nº 7.347/85.",
    ]

    for texto in exemplos:
        inferencia(texto, modelo_ner)

    log.info("=" * 70)
    log.info("PIPELINE CONCLUÍDO!")
    log.info(f"Modelo salvo em: {DIR_NER}")
    log.info("=" * 70)