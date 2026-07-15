"""
Pipeline de Curadoria do Corpus Ulysses Tesemô — Versão em Lotes
=================================================================
Autor: Bruno Martins Costa (TCC - UFCAT 2025)
Orientador: Prof. Dr. Marcio de Souza Dias

Otimizado para Ryzen 7 5700X (8 núcleos / 16 threads)
Processa em lotes de 50k arquivos para não estourar a RAM.

Requisitos:
  pip install langdetect datasketch tqdm

Como usar:
  python tesemo_pipeline.py              # roda tudo
  python tesemo_pipeline.py --analisar   # resumo por categoria após rodar
  python tesemo_pipeline.py --workers 8  # força número de workers
  python tesemo_pipeline.py --batch 30000 # tamanho do lote (padrão 50000)
"""

import os
import re
import csv
import sys
import hashlib
import logging
import unicodedata
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial

# ── Dependências opcionais ──────────────────────────────────────────────────
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_OK = True
except ImportError:
    MINHASH_OK = False

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES
# ════════════════════════════════════════════════════════════════════════════

INPUT_DIR     = Path("./tesemo_raw")
OUTPUT_DIR    = Path("./tesemo_clean")
REPORT_CSV    = Path("./tesemo_relatorio.csv")
CHECKPOINT    = Path("./tesemo_checkpoint.txt")

TARGET_LANG   = "pt"
MIN_CHARS     = 200
LSH_THRESHOLD = 0.85
NUM_PERM      = 128
SHINGLE_SIZE  = 5
BATCH_SIZE    = 50_000   # arquivos por lote — ajuste se necessário

DEFAULT_WORKERS = max(1, mp.cpu_count() - 2)  # 14 no 5700X

CATEGORIAS_SEGURAS = {
    "J1","J2","L1","L2","L3","L4","L5","L6","L7","L8","L9",
    "L10","L11","L12","L13","A2","O1","O2","O3","O6","O7",
    "O8","O9","O10","O11","O12","O13",
}

# ════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.lower()
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def hash_documento(texto_norm: str) -> str:
    return hashlib.sha256(texto_norm.encode("utf-8")).hexdigest()


def get_shingles(texto: str, k: int = SHINGLE_SIZE) -> set:
    return {texto[i:i+k] for i in range(len(texto) - k + 1)}


def minhash_de_texto(texto_norm: str) -> "MinHash":
    m = MinHash(num_perm=NUM_PERM)
    for shingle in get_shingles(texto_norm):
        m.update(shingle.encode("utf8"))
    return m


def categoria_do_path(caminho: Path, input_dir: Path) -> str:
    relativo = caminho.relative_to(input_dir)
    partes = relativo.parts
    return partes[0].upper() if partes else "DESCONHECIDO"


# ════════════════════════════════════════════════════════════════════════════
#  WORKER — processa UM arquivo e retorna resultado leve (sem texto completo)
# ════════════════════════════════════════════════════════════════════════════

def processar_arquivo(caminho_str: str, input_dir_str: str) -> dict:
    """
    Executado em paralelo. Retorna dict LEVE:
    - texto_norm só é retornado se o arquivo passar nos filtros de tamanho/idioma
    - texto_norm é descartado após calcular o hash para não sobrecarregar a fila IPC
    """
    caminho   = Path(caminho_str)
    input_dir = Path(input_dir_str)
    categoria = categoria_do_path(caminho, input_dir)

    resultado = {
        "arquivo":     str(caminho.relative_to(input_dir)),
        "categoria":   categoria,
        "status":      "",
        "idioma":      "",
        "hash":        "",
        "minhash_data": None,   # só preenchido se passar em tudo
        "caminho_abs": caminho_str,
    }

    # ── Leitura ──────────────────────────────────────────────────────────────
    try:
        texto = caminho.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            texto = caminho.read_text(encoding="latin-1")
        except Exception:
            resultado["status"] = "erro_leitura"
            return resultado

    # ── Filtro 1: tamanho mínimo ──────────────────────────────────────────────
    if len(texto) < MIN_CHARS:
        resultado["status"] = "rejeitado_curto"
        return resultado

    # ── Filtro 2: idioma ──────────────────────────────────────────────────────
    if categoria in CATEGORIAS_SEGURAS:
        lang = "pt"
    elif LANGDETECT_OK:
        try:
            lang = detect(texto[:2000])
        except LangDetectException:
            lang = "erro"
    else:
        lang = "desconhecido"

    resultado["idioma"] = lang

    if lang not in ("pt", "desconhecido") and categoria not in CATEGORIAS_SEGURAS:
        resultado["status"] = f"rejeitado_idioma_{lang}"
        return resultado

    # ── Normalização + hash + minhash ─────────────────────────────────────────
    texto_norm = normalizar_texto(texto)
    del texto  # libera memória imediatamente

    resultado["hash"] = hash_documento(texto_norm)

    # Calcula MinHash aqui no worker (paralelo) e passa só o hashvalues (array numpy pequeno)
    if MINHASH_OK:
        mh = minhash_de_texto(texto_norm)
        resultado["minhash_data"] = mh.hashvalues.tolist()  # lista de ints, muito menor que o texto

    del texto_norm  # libera memória

    resultado["status"] = "pendente"
    return resultado


# ════════════════════════════════════════════════════════════════════════════
#  DEDUPLICAÇÃO — serial, opera sobre resultados de um lote
#  Recebe hashes_vistos e lsh de fora para acumular entre lotes
# ════════════════════════════════════════════════════════════════════════════

def deduplicar_lote(resultados: list[dict],
                    hashes_vistos: set,
                    lsh,
                    minhash_idx_start: int) -> tuple[list[dict], int]:
    """
    Deduplica um lote. Modifica resultados in-place.
    Retorna (resultados, novo_minhash_idx).
    """
    minhash_idx = minhash_idx_start

    for r in resultados:
        if r["status"] != "pendente":
            continue

        # Dedup exata
        if r["hash"] in hashes_vistos:
            r["status"] = "duplicata_exata"
            continue
        hashes_vistos.add(r["hash"])

        # Dedup fuzzy
        if MINHASH_OK and lsh is not None and r["minhash_data"] is not None:
            import numpy as np
            mh = MinHash(num_perm=NUM_PERM)
            mh.hashvalues = np.array(r["minhash_data"], dtype=np.uint64)
            if lsh.query(mh):
                r["status"] = "duplicata_fuzzy"
                r["minhash_data"] = None
                continue
            lsh.insert(f"doc_{minhash_idx}", mh)
            minhash_idx += 1

        r["status"] = "aceito"
        r["minhash_data"] = None  # libera memória

    return resultados, minhash_idx


# ════════════════════════════════════════════════════════════════════════════
#  CÓPIA — copia um arquivo aceito para OUTPUT_DIR
# ════════════════════════════════════════════════════════════════════════════

def copiar_arquivo(item: dict, output_dir_str: str) -> None:
    if item["status"] != "aceito":
        return
    src  = Path(item["caminho_abs"])
    dest = Path(output_dir_str) / item["arquivo"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        texto = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        texto = src.read_text(encoding="latin-1")
    dest.write_text(texto, encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def pipeline(num_workers: int = DEFAULT_WORKERS, batch_size: int = BATCH_SIZE):
    if not INPUT_DIR.exists():
        log.error(f"Pasta não encontrada: {INPUT_DIR}")
        return

    if not LANGDETECT_OK:
        log.warning("langdetect não instalado — filtro de idioma desativado.")
    if not MINHASH_OK:
        log.warning("datasketch não instalado — dedup fuzzy desativada.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Coleta arquivos e aplica checkpoint ────────────────────────────────
    log.info("Varrendo arquivos...")
    todos = sorted(str(p) for p in INPUT_DIR.rglob("*.txt"))

    ja_processados: set[str] = set()
    if CHECKPOINT.exists():
        ja_processados = set(CHECKPOINT.read_text(encoding="utf-8").splitlines())
        log.info(f"Checkpoint: {len(ja_processados):,} já processados — retomando.")

    pendentes = [p for p in todos if p not in ja_processados]
    n_lotes   = (len(pendentes) + batch_size - 1) // batch_size

    log.info(f"Total: {len(todos):,} | Pendentes: {len(pendentes):,} | "
             f"Lotes de {batch_size:,}: {n_lotes}")
    log.info(f"Workers: {num_workers} | RAM por lote: ~{batch_size * 0.002:.0f} MB estimado")

    if not pendentes:
        log.info("Nada a processar.")
        return

    # ── Estado acumulado entre lotes (dedup cross-lote) ───────────────────
    hashes_vistos: set[str] = set()
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM) if MINHASH_OK else None
    minhash_idx = 0

    worker_fn = partial(processar_arquivo, input_dir_str=str(INPUT_DIR))
    copiar_fn = partial(copiar_arquivo, output_dir_str=str(OUTPUT_DIR))

    # ── Abre o relatório CSV em modo append ───────────────────────────────
    csv_novo = not REPORT_CSV.exists()
    csv_file = REPORT_CSV.open("a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file,
                                fieldnames=["arquivo","categoria","status","idioma"])
    if csv_novo:
        csv_writer.writeheader()

    # ── Loop por lotes ─────────────────────────────────────────────────────
    total_aceitos = total_rejeitados = 0

    for n_lote, inicio in enumerate(range(0, len(pendentes), batch_size), start=1):
        lote = pendentes[inicio : inicio + batch_size]
        log.info(f"Lote {n_lote}/{n_lotes} — {len(lote):,} arquivos")

        # FASE 1: paralela
        resultados = []
        with mp.Pool(processes=num_workers) as pool:
            for r in tqdm(
                pool.imap_unordered(worker_fn, lote, chunksize=100),
                total=len(lote),
                desc=f"  Fase1 lote {n_lote}",
                unit="arq",
            ):
                resultados.append(r)

        # FASE 2: deduplicação serial
        resultados, minhash_idx = deduplicar_lote(
            resultados, hashes_vistos, lsh, minhash_idx
        )

        # FASE 3: cópia paralela dos aceitos
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(copiar_fn, resultados, chunksize=100),
                total=len(resultados),
                desc=f"  Fase3 lote {n_lote}",
                unit="arq",
            ))

        # Salva no CSV e checkpoint
        aceitos_lote = 0
        for r in resultados:
            csv_writer.writerow({
                "arquivo":   r["arquivo"],
                "categoria": r["categoria"],
                "status":    r["status"],
                "idioma":    r["idioma"],
            })
            if r["status"] == "aceito":
                aceitos_lote += 1
        csv_file.flush()

        with CHECKPOINT.open("a", encoding="utf-8") as f:
            for r in resultados:
                f.write(r["caminho_abs"] + "\n")

        rej_lote = len(resultados) - aceitos_lote
        total_aceitos     += aceitos_lote
        total_rejeitados  += rej_lote
        processados_ate_agora = total_aceitos + total_rejeitados
        taxa = total_aceitos / processados_ate_agora * 100

        log.info(f"  Lote {n_lote} concluído — "
                 f"aceitos: {aceitos_lote:,} | rejeitados: {rej_lote:,} | "
                 f"taxa acum.: {taxa:.1f}%")

        del resultados  # libera RAM do lote

    csv_file.close()

    # ── Resumo final ──────────────────────────────────────────────────────
    grand_total = total_aceitos + total_rejeitados
    print("\n" + "="*58)
    print("  RESUMO FINAL")
    print("="*58)
    print(f"  Total processado  : {grand_total:>10,}")
    print(f"  Aceitos           : {total_aceitos:>10,}")
    print(f"  Rejeitados/dup    : {total_rejeitados:>10,}")
    print(f"  Taxa aproveitamento: {total_aceitos/grand_total*100:>8.1f}%")
    print("="*58)
    print(f"  Arquivos limpos: {OUTPUT_DIR}")
    print(f"  Relatório      : {REPORT_CSV}")
    print("="*58)


# ════════════════════════════════════════════════════════════════════════════
#  ANÁLISE DO RELATÓRIO
# ════════════════════════════════════════════════════════════════════════════

def analisar_relatorio():
    if not REPORT_CSV.exists():
        print("Relatório não encontrado. Rode o pipeline primeiro.")
        return

    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(int))

    with REPORT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats[row["categoria"]][row["status"]] += 1

    print(f"\n{'CATEGORIA':<12} {'ACEITOS':>10} {'REJEITADOS':>12} {'TOTAL':>8}")
    print("-" * 48)
    grand_a = grand_r = 0
    for cat in sorted(stats):
        a = stats[cat].get("aceito", 0)
        r = sum(v for k,v in stats[cat].items() if k != "aceito")
        grand_a += a
        grand_r += r
        print(f"{cat:<12} {a:>10,} {r:>12,} {a+r:>8,}")
    print("-" * 48)
    gt = grand_a + grand_r
    print(f"{'TOTAL':<12} {grand_a:>10,} {grand_r:>12,} {gt:>8,}")
    if gt:
        print(f"\nTaxa geral: {grand_a/gt*100:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.freeze_support()

    workers    = DEFAULT_WORKERS
    batch_size = BATCH_SIZE

    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass

    if "--batch" in sys.argv:
        idx = sys.argv.index("--batch")
        try:
            batch_size = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass

    if "--analisar" in sys.argv:
        analisar_relatorio()
    else:
        print(f"Iniciando pipeline com {workers} workers, lotes de {batch_size:,}...")
        print(f"Processador detectado: {mp.cpu_count()} threads disponíveis")
        print(f"Entrada : {INPUT_DIR}")
        print(f"Saída   : {OUTPUT_DIR}\n")
        pipeline(num_workers=workers, batch_size=batch_size)
