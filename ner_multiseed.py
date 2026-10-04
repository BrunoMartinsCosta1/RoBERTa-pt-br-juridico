r"""
ner_multiseed.py — NER no LeNER-Br com várias sementes, com e sem DAPT.

NÃO altera nada em experimentos\bert nem experimentos\roberta (só LÊ de
experimentos\{modelo}\mlm e experimentos\{modelo}\corpus\lener_br).
Tudo que gera vai para experimentos\multiseed\.

Hiperparâmetros de NER idênticos aos do jurisroberta_pipeline.py
(5 épocas, batch 16 x acc 2, warmup 0.1, wd 0.01, bf16, best-val-F1).

USO (PowerShell, na pasta E:\Pfc2, venv ativado):
  python -u .\ner_multiseed.py rodar 2>&1 | Tee-Object -FilePath experimentos\multiseed_console.txt
  python .\ner_multiseed.py agregar

Opções de "rodar":
  --modelos bert roberta        (padrão: os dois)
  --configs dapt base           (padrão: os dois; base = checkpoint HF sem DAPT)
  --seeds 42 43 44 45 46        (padrão)

É retomável: se cair, rode de novo o mesmo comando; execuções já concluídas
(resultado.json existente) são puladas.
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

HF = {"bert": "neuralmind/bert-base-portuguese-cased", "roberta": "xlm-roberta-base"}
RAIZ = Path("experimentos")
SAIDA = RAIZ / "multiseed"
SAIDA.mkdir(parents=True, exist_ok=True)
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

MAX_SEQ_LENGTH = 512
BATCH_NER, GRAD_ACC_NER, EPOCHS_NER = 16, 2, 5
WORKERS = 0  # 0 evita travamento de DataLoader multiprocess no Windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(SAIDA / "multiseed.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("multiseed")


def ler_conll(caminho):
    toks, tgs, t, g = [], [], [], []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                if t:
                    toks.append(t); tgs.append(g); t, g = [], []
            else:
                p = linha.split()
                t.append(p[0]); g.append(p[-1])
    if t:
        toks.append(t); tgs.append(g)
    return toks, tgs


def preparar_dados(modelo, tokenizer):
    """Tokeniza LeNER-Br uma vez por modelo (cache Arrow em multiseed/cache)."""
    from datasets import Dataset, DatasetDict, load_from_disk
    dir_lener = RAIZ / modelo / "corpus" / "lener_br"
    brutos, tags_all = {}, set()
    for split in ["train", "validation", "test"]:
        toks, tgs = ler_conll(dir_lener / f"{split}.conll")
        brutos[split] = (toks, tgs)
        for g in tgs:
            tags_all.update(g)
    LABELS = ["O"] + sorted(t for t in tags_all if t != "O")  # mesma ordem do pipeline
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    cache = SAIDA / "cache" / modelo
    if cache.exists() and any(cache.iterdir()):
        return load_from_disk(str(cache)), LABELS, label2id, id2label

    ds = DatasetDict({s: Dataset.from_dict({"tokens": t, "ner_tags": g})
                      for s, (t, g) in brutos.items()})

    def tok(ex):
        enc = tokenizer(ex["tokens"], truncation=True, max_length=MAX_SEQ_LENGTH,
                        is_split_into_words=True, padding=False)
        out = []
        for i, tags in enumerate(ex["ner_tags"]):
            prev, ids = None, []
            for wid in enc.word_ids(batch_index=i):
                if wid is None:
                    ids.append(-100)
                elif wid != prev:
                    ids.append(label2id.get(tags[wid], 0) if wid < len(tags) else -100)
                else:
                    ids.append(-100)
                prev = wid
            out.append(ids)
        enc["labels"] = out
        return enc

    tokd = ds.map(tok, batched=True, num_proc=4,
                  remove_columns=ds["train"].column_names, desc=f"Tokenizando {modelo}")
    tokd.save_to_disk(str(cache))
    return tokd, LABELS, label2id, id2label


def rodar_uma(modelo, config, seed, dados, tokenizer, LABELS, label2id, id2label):
    import numpy as np
    import torch
    from seqeval.metrics import classification_report, f1_score
    from transformers import (AutoModelForTokenClassification,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments, set_seed)

    nome = f"{modelo}_{config}_seed{seed}"
    pasta = SAIDA / nome
    if (pasta / "resultado.json").exists():
        log.info(f"[{nome}] já concluído. Pulando.")
        return
    if pasta.exists():
        shutil.rmtree(pasta)  # execução incompleta: recomeça do zero
    pasta.mkdir(parents=True)

    origem = str(RAIZ / modelo / "mlm") if config == "dapt" else HF[modelo]
    log.info(f"[{nome}] início — modelo inicial: {origem}")
    set_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        origem, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)
    model.gradient_checkpointing_enable()

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True,
                                                  pad_to_multiple_of=8)

    def pares(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        pv = [[id2label[p] for p, l in zip(pr, la) if l != -100] for pr, la in zip(preds, labels)]
        lv = [[id2label[l] for p, l in zip(pr, la) if l != -100] for pr, la in zip(preds, labels)]
        return lv, pv

    def metricas(pred):
        lv, pv = pares(pred)
        rel = classification_report(lv, pv, output_dict=True, zero_division=0)
        res = {"f1": f1_score(lv, pv)}
        for ent, v in rel.items():
            if isinstance(v, dict) and "f1-score" in v:
                res[f"f1_{ent.lower()}"] = v["f1-score"]
        return res

    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(pasta / "tmp"),
        num_train_epochs=EPOCHS_NER,
        per_device_train_batch_size=BATCH_NER, per_device_eval_batch_size=BATCH_NER,
        gradient_accumulation_steps=GRAD_ACC_NER,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        fp16=cuda and not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, warmup_ratio=0.1, weight_decay=0.01, disable_tqdm=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=WORKERS, dataloader_pin_memory=True,
        torch_compile=False, report_to="none", seed=seed, data_seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dados["train"],
                      eval_dataset=dados["validation"], processing_class=tokenizer,
                      data_collator=collator, compute_metrics=metricas)
    trainer.train()
    trainer.save_state()  # trainer_state.json completo (log_history de todas as épocas)
    shutil.copy(pasta / "tmp" / "trainer_state.json", pasta / "trainer_state.json")

    pred = trainer.predict(dados["test"], metric_key_prefix="eval")
    res = {k: float(v) for k, v in pred.metrics.items() if isinstance(v, (int, float))}
    lv, pv = pares((pred.predictions, pred.label_ids))
    with open(pasta / "predicoes_teste.json", "w", encoding="utf-8") as f:
        json.dump({"gold": lv, "pred": pv}, f)

    estado = json.load(open(pasta / "trainer_state.json", encoding="utf-8"))
    res["melhor_epoca_val_f1"] = max((e["eval_f1"], e["epoch"]) for e in estado["log_history"]
                                     if "eval_f1" in e)[1]
    res["melhor_val_f1"] = max(e["eval_f1"] for e in estado["log_history"] if "eval_f1" in e)
    res.update(modelo=modelo, config=config, seed=seed)

    shutil.rmtree(pasta / "tmp", ignore_errors=True)  # apaga pesos (~400 MB-1 GB)
    with open(pasta / "resultado.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log.info(f"[{nome}] TESTE F1={res['eval_f1']:.4f} | melhor val F1={res['melhor_val_f1']:.4f} "
             f"(época {res['melhor_epoca_val_f1']})")
    del trainer, model
    if cuda:
        torch.cuda.empty_cache()


def cmd_rodar(a):
    from transformers import AutoTokenizer
    for modelo in a.modelos:
        # tokenizador do DAPT == tokenizador do modelo base (MLM não altera vocabulário)
        tokenizer = AutoTokenizer.from_pretrained(str(RAIZ / modelo / "mlm"))
        dados, LABELS, l2i, i2l = preparar_dados(modelo, tokenizer)
        for config in a.configs:
            for seed in a.seeds:
                rodar_uma(modelo, config, seed, dados, tokenizer, LABELS, l2i, i2l)
    log.info("TUDO CONCLUÍDO. Agora: python .\\ner_multiseed.py agregar")


def cmd_agregar(_):
    import numpy as np
    from scipy import stats
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(SAIDA.glob("*/resultado.json"))]
    if not runs:
        print("Nenhum resultado encontrado."); return
    chaves = ["eval_f1", "eval_f1_macro avg", "eval_f1_jurisprudencia", "eval_f1_legislacao",
              "eval_f1_local", "eval_f1_organizacao", "eval_f1_pessoa", "eval_f1_tempo",
              "melhor_val_f1"]
    grupos = {}
    for r in runs:
        grupos.setdefault((r["modelo"], r["config"]), []).append(r)

    linhas = ["# Resultados NER — média ± desvio-padrão (ddof=1) sobre sementes\n",
              "| modelo | config | n | " + " | ".join(k.replace("eval_f1_", "").replace("eval_", "")
                                                       for k in chaves) + " |",
              "|" + "---|" * (len(chaves) + 3)]
    for (m, c), rs in sorted(grupos.items()):
        cel = []
        for k in chaves:
            v = np.array([x[k] for x in rs]) * 100
            cel.append(f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}")
        linhas.append(f"| {m} | {c} | {len(rs)} | " + " | ".join(cel) + " |")

    linhas += ["\n## F1 de teste por semente\n", "| modelo | config | semente | F1 teste | época do melhor val |",
               "|---|---|---|---|---|"]
    for (m, c), rs in sorted(grupos.items()):
        for r in sorted(rs, key=lambda x: x["seed"]):
            linhas.append(f"| {m} | {c} | {r['seed']} | {r['eval_f1']*100:.2f} | {int(r['melhor_epoca_val_f1'])} |")

    def comp(a, b, rot):
        if a in grupos and b in grupos:
            x = np.array([r["eval_f1"] for r in grupos[a]]) * 100
            y = np.array([r["eval_f1"] for r in grupos[b]]) * 100
            if len(x) > 1 and len(y) > 1:
                t, p = stats.ttest_ind(x, y, equal_var=False)
                u, pu = stats.mannwhitneyu(x, y, alternative="two-sided")
                linhas.append(f"| {rot} | {x.mean()-y.mean():+.2f} p.p. | {p:.4f} | {pu:.4f} |")

    linhas += ["\n## Comparações (diferença de médias de F1 de teste)\n",
               "| comparação | Δ | p (Welch) | p (Mann-Whitney) |", "|---|---|---|---|"]
    for m in ("bert", "roberta"):
        comp((m, "dapt"), (m, "base"), f"{m}: DAPT − sem DAPT")
    comp(("bert", "dapt"), ("roberta", "dapt"), "DAPT: BERTimbau − XLM-R")
    comp(("bert", "base"), ("roberta", "base"), "sem DAPT: BERTimbau − XLM-R")
    linhas.append("\nCom n=5 por grupo o poder é baixo; com Mann-Whitney o menor p possível é 0,0079.")
    saida = SAIDA / "resumo_multiseed.md"
    saida.write_text("\n".join(linhas), encoding="utf-8")
    print("\n".join(linhas)); print(f"\n(salvo em {saida})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("rodar")
    r.add_argument("--modelos", nargs="+", default=["bert", "roberta"], choices=["bert", "roberta"])
    r.add_argument("--configs", nargs="+", default=["dapt", "base"], choices=["dapt", "base"])
    r.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    r.set_defaults(f=cmd_rodar)
    g = sp.add_parser("agregar"); g.set_defaults(f=cmd_agregar)
    a = ap.parse_args(); a.f(a)
