from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


BEH_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def read_news_tsv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        quoting=3,
        dtype=str,
    )
    for c in ["category", "subcategory", "title", "abstract"]:
        df[c] = df[c].fillna("")
    df["text"] = (
        df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")
    ).str.strip()
    return df


def parse_impressions(impr: str) -> tuple[list[str], list[int]]:
    # Format: "N12345-1 N54321-0 ..."
    items = []
    labels = []
    if not isinstance(impr, str) or not impr.strip():
        return items, labels
    for tok in impr.strip().split():
        if "-" in tok:
            nid, lab = tok.rsplit("-", 1)
            items.append(nid)
            labels.append(int(lab))
        else:
            items.append(tok)
            labels.append(0)
    return items, labels


def time_feature_indices(time_value: object) -> tuple[int, int]:
    """Return (hour_idx, weekday_idx) from a MIND behavior timestamp.

    Weekdays use pandas/Python convention: Monday=0, ..., Sunday=6.
    """
    parsed = pd.to_datetime(time_value, format=MIND_TIME_FORMAT, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse MIND behavior timestamp: {time_value!r}")
    ts = pd.Timestamp(parsed)
    return int(ts.hour), int(ts.dayofweek)


def read_behaviors_tsv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEH_COLUMNS,
        quoting=3,
        dtype=str,
    )
    df["history"] = df["history"].fillna("").apply(lambda s: s.split() if s else [])
    parsed = df["impressions"].fillna("").apply(parse_impressions)
    df["cand_news_id"] = parsed.apply(lambda x: x[0])
    df["cand_label"] = parsed.apply(lambda x: x[1])
    return df


def iter_behaviors_tsv(path: str | Path) -> Iterator[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            impression_id, user_id, time, history, impressions = parts[:5]
            cand_news_id, cand_label = parse_impressions(impressions)
            yield {
                "impression_id": impression_id,
                "user_id": user_id,
                "time": time,
                "history": history.split() if history else [],
                "cand_news_id": cand_news_id,
                "cand_label": cand_label,
            }


def count_behavior_rows(path: str | Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def sub_sample_behaviors(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
