from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from tqdm import tqdm


def _parse_entity_column(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        rows = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("WikidataId") or "").strip()
        if entity_id:
            ids.append(entity_id)
    return ids


def _read_mind_entities(news_paths: Iterable[Path]) -> set[str]:
    entities: set[str] = set()
    for news_path in news_paths:
        if not news_path.exists():
            continue
        news = pd.read_csv(
            news_path,
            sep="\t",
            header=None,
            usecols=[6, 7],
            names=["title_entities", "abstract_entities"],
            quoting=3,
            dtype=str,
        )
        for row in tqdm(
            news.itertuples(index=False),
            total=len(news),
            desc=f"Read entities ({news_path.parent.name})",
        ):
            entities.update(_parse_entity_column(row.title_entities))
            entities.update(_parse_entity_column(row.abstract_entities))
    return entities


def _read_ids_from_vec(paths: list[Path] | None) -> set[str] | None:
    if not paths:
        return None
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    ids.add(parts[0])
    return ids


def _iter_triple_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".txt", ".tsv", ".gz", ".parquet"}
    return [
        p
        for p in sorted(path.rglob("*"))
        if p.is_file() and (p.suffix.lower() in suffixes or p.name.endswith(".txt.gz"))
    ]


def _iter_text_triples(path: Path) -> Iterator[tuple[str, str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            yield parts[0], parts[1], parts[2]


def _iter_parquet_triples(path: Path) -> Iterator[tuple[str, str, str]]:
    df = pd.read_parquet(path)
    candidates = [
        ("head", "relation", "tail"),
        ("subject", "predicate", "object"),
    ]
    for cols in candidates:
        if all(col in df.columns for col in cols):
            for row in df[list(cols)].itertuples(index=False, name=None):
                yield str(row[0]), str(row[1]), str(row[2])
            return
    if len(df.columns) >= 3:
        cols = list(df.columns[:3])
        for row in df[cols].itertuples(index=False, name=None):
            yield str(row[0]), str(row[1]), str(row[2])


def _iter_triples(path: Path) -> Iterator[tuple[str, str, str]]:
    if path.suffix.lower() == ".parquet":
        yield from _iter_parquet_triples(path)
    else:
        yield from _iter_text_triples(path)


def build_triples(
    kg_path: Path,
    seed_entities: set[str],
    output_path: Path,
    entity_vocab: set[str] | None,
    relation_vocab: set[str] | None,
    max_triples_per_seed: int,
    keep_tail_without_embedding: bool,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_by_seed: dict[str, int] = {}
    seeds_with_triples: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    n_seen = 0
    n_kept = 0
    n_files = 0

    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        for triple_file in _iter_triple_files(kg_path):
            n_files += 1
            for head, relation, tail in tqdm(
                _iter_triples(triple_file),
                desc=f"Filter {triple_file.name}",
                unit=" triples",
            ):
                n_seen += 1
                head_is_seed = head in seed_entities
                tail_is_seed = tail in seed_entities
                if not head_is_seed and not tail_is_seed:
                    continue
                if relation_vocab is not None and relation not in relation_vocab:
                    continue
                if entity_vocab is not None:
                    if head not in entity_vocab:
                        continue
                    if tail not in entity_vocab and not keep_tail_without_embedding:
                        continue

                seed = head if head_is_seed else tail
                seeds_with_triples.add(seed)
                if max_triples_per_seed > 0:
                    current = counts_by_seed.get(seed, 0)
                    if current >= max_triples_per_seed:
                        continue
                    counts_by_seed[seed] = current + 1

                triple = (head, relation, tail)
                if triple in seen:
                    continue
                seen.add(triple)
                out.write(f"{head}\t{relation}\t{tail}\n")
                n_kept += 1

    return {
        "n_files": n_files,
        "n_input_triples_scanned": n_seen,
        "n_seed_entities": len(seed_entities),
        "n_seed_entities_with_triples": len(seeds_with_triples),
        "n_output_triples": n_kept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a Wikidata5M-style KG down to one-hop triples touching entities "
            "mentioned in MIND news.tsv."
        )
    )
    parser.add_argument("--kg-path", required=True, type=Path)
    parser.add_argument("--raw-root", default="data/raw", type=Path)
    parser.add_argument("--train-dir", default="MINDsmall_train")
    parser.add_argument("--dev-dir", default="MINDsmall_dev")
    parser.add_argument(
        "--output",
        default="data/processed/MINDsmall/kg_triples.tsv",
        type=Path,
    )
    parser.add_argument(
        "--entity-embedding",
        default=None,
        action="append",
        type=Path,
        help=(
            "Optional entity_embedding.vec used to keep only embeddable entities. "
            "Repeat to union train/dev embedding vocabularies."
        ),
    )
    parser.add_argument(
        "--relation-embedding",
        default=None,
        action="append",
        type=Path,
        help=(
            "Optional relation_embedding.vec used to keep only embeddable relations. "
            "Repeat to union train/dev embedding vocabularies."
        ),
    )
    parser.add_argument(
        "--max-triples-per-seed",
        default=100,
        type=int,
        help="Limit triples retained per MIND entity. Use 0 for no limit.",
    )
    parser.add_argument(
        "--keep-tail-without-embedding",
        action="store_true",
        help="Keep triples whose non-MIND neighbor has no entity embedding.",
    )
    args = parser.parse_args()

    news_paths = [
        args.raw_root / args.train_dir / "news.tsv",
        args.raw_root / args.dev_dir / "news.tsv",
    ]
    seed_entities = _read_mind_entities(news_paths)
    entity_vocab = _read_ids_from_vec(args.entity_embedding)
    relation_vocab = _read_ids_from_vec(args.relation_embedding)
    stats = build_triples(
        kg_path=args.kg_path,
        seed_entities=seed_entities,
        output_path=args.output,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        max_triples_per_seed=int(args.max_triples_per_seed),
        keep_tail_without_embedding=bool(args.keep_tail_without_embedding),
    )
    stats_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote triples to {args.output}")
    print(f"Wrote metadata to {stats_path}")


if __name__ == "__main__":
    main()
