"""YAML rule evaluation.

Rules are config, not code — you will want fifteen variants of one rule before
you want a second rule. Expressions evaluate against feature-frame columns via
``df.eval``, and every hit carries the values that triggered it so it can be
audited after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_MIN_TURNOVER = 10_000_000.0   # 1 crore median 20-day turnover
DEFAULT_MIN_PRICE = 20.0


class RuleError(Exception):
    pass


@dataclass
class Rule:
    name: str
    expr: str
    title: str = ""
    description: str = ""
    horizon: str = ""       # swing | positional | intraday-adjacent
    origin: str = ""        # where the setup comes from
    why: str = ""           # what the conditions are trying to capture
    manage: str = ""        # how the setup is conventionally traded
    min_turnover: float = DEFAULT_MIN_TURNOVER
    min_price: float = DEFAULT_MIN_PRICE
    min_history: int = 200
    emit: list[str] = field(default_factory=list)
    sort_by: str = ""
    ascending: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        missing = [k for k in ("name", "expr") if k not in d]
        if missing:
            raise RuleError(f"rule missing {missing}: {d}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise RuleError(f"rule {d['name']} has unknown keys: {sorted(unknown)}")
        d = dict(d)
        # YAML block scalars keep their newlines, and pandas.eval parses a
        # newline inside parentheses as an unterminated statement. Rules are
        # written across several lines to stay readable, so the expression is
        # flattened here rather than every author having to remember.
        d["expr"] = " ".join(str(d["expr"]).split())
        return cls(**d)


def load_rules(path: str | Path) -> list[Rule]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list):
        raise RuleError("rules file must be a list of rules")
    rules = [Rule.from_dict(r) for r in raw]
    names = [r.name for r in rules]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise RuleError(f"duplicate rule names: {sorted(dupes)}")
    return rules


def liquidity_mask(df: pd.DataFrame, rule: Rule) -> pd.Series:
    """Median 20-day turnover floor plus a minimum price.

    Without this the hit list fills with untradeable microcaps.
    """
    turnover = df.get("turnover_median_20d")
    price = df.get("close")
    if price is None or price.isna().all():
        price = df.get("adj_close")
    mask = pd.Series(True, index=df.index)
    if turnover is not None:
        mask &= turnover.fillna(0) >= rule.min_turnover
    if price is not None:
        mask &= price.fillna(0) >= rule.min_price
    return mask


def evaluate_rule(features: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """Rows of ``features`` matching ``rule``, with the triggering values."""
    df = features
    if df.empty:
        return df

    eligible = liquidity_mask(df, rule)
    if "bars_available" in df.columns:
        eligible &= df["bars_available"].fillna(0) >= rule.min_history
    # A row whose indicators span an unadjusted corporate action is not a
    # near-miss, it is a number computed across a discontinuity. Excluded
    # outright rather than ranked against clean ones.
    if "contaminated" in df.columns:
        eligible &= ~df["contaminated"].fillna(False).astype(bool)

    try:
        matched = df.eval(rule.expr)
    except Exception as exc:
        raise RuleError(f"rule {rule.name}: cannot evaluate {rule.expr!r}: {exc}") from exc
    if not isinstance(matched, pd.Series) or matched.dtype != bool:
        raise RuleError(f"rule {rule.name}: expr must produce a boolean, got {rule.expr!r}")

    # A NaN comparison is False, never a hit. Insufficient history must filter
    # out rather than quietly pass.
    hits = df[eligible & matched.fillna(False)].copy()
    if hits.empty:
        return hits
    hits.insert(0, "rule", rule.name)
    if rule.sort_by and rule.sort_by in hits.columns:
        hits = hits.sort_values(rule.sort_by, ascending=rule.ascending)
    return hits.reset_index(drop=True)


def run_rules(features: pd.DataFrame, rules: list[Rule]) -> dict[str, pd.DataFrame]:
    return {r.name: evaluate_rule(features, r) for r in rules}
