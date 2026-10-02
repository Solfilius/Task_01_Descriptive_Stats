#!/usr/bin/env python3
"""
pure_python_stats.py -- descriptive statistics using ONLY the standard library.

Usage
-----
    python pure_python_stats.py [path/to/file.csv] [--top N] [--json-out results.json]

If no path is given, the script looks for data/fb_ads_president_scored_anon.csv
relative to the current working directory.

What it reports
---------------
Dataset level : row count, column count, missing values per column, and an
                inferred data type per column.
Numeric cols  : count, mean, min, max, standard deviation, median.
Other cols    : count, number of unique values, mode (+ frequency), top-N values.

Design decisions (these are the places where Pandas would decide silently)
--------------------------------------------------------------------------
1. MISSING VALUES. A cell is "missing" if it is empty, whitespace-only, or one
   of the tokens in MISSING_TOKENS (which mirrors Pandas' default NA strings, so
   the two scripts can be compared fairly in Phase 2). Missing cells are
   excluded from every statistic, never treated as zero.

2. TYPE INFERENCE. Every distinct non-missing value is classified as number,
   date (YYYY-MM-DD), range (a stringified dict with 'lower_bound' and
   optionally 'upper_bound'), list (a stringified Python list), or text. A
   column takes the type of its dominant class if that class covers at least
   TYPE_THRESHOLD of its non-missing values; otherwise it is treated as text.
   Entries that do not fit the winning type are counted and reported, not
   silently dropped. Currency symbols ($, GBP, EUR, JPY signs) and thousands
   separators ("$1,234.56") are accepted when parsing numbers.

3. RANGE COLUMNS. The Facebook Ad Library does not publish exact spend or
   impressions, only bins such as {'lower_bound': '100', 'upper_bound': '199'}.
   Such a column is therefore summarised two ways: as a categorical column
   (which bins are most common) and numerically on three derived series:
   lower_bound, upper_bound, and midpoint. Open-ended top bins (no upper
   bound) have no upper_bound or midpoint, so those series have a smaller
   count; the number of open-ended bins is reported.

4. STANDARD DEVIATION is the SAMPLE standard deviation (divide by n - 1), the
   same default as pandas' Series.std(). It is undefined (shown as n/a) when
   there are fewer than two values.

5. MODE TIES are broken by sorting tied values alphabetically, so output is
   deterministic. Pandas may order ties differently.
"""

import argparse
import ast
import csv
import datetime
import json
import math
import os
import re
import sys
from collections import Counter

DEFAULT_CSV = os.path.join("data", "fb_ads_president_scored_anon.csv")
DEFAULT_TOP_N = 5
TYPE_THRESHOLD = 0.95   # share of non-missing values a type must cover to win
DISPLAY_WIDTH = 70      # long values (e.g. hashes) are shortened when printed

# Mirrors the default NA strings used by pandas.read_csv (case-sensitive).
MISSING_TOKENS = {
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
}

CURRENCY_TABLE = {ord(ch): None for ch in "$\u00a3\u20ac\u00a5"}
NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
INTEGER_RE = re.compile(r"^[+-]?\d+$")
THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d*)?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------
# Value parsing helpers
# --------------------------------------------------------------------------

def is_missing(raw):
    """True if a raw CSV cell should be treated as missing."""
    stripped = raw.strip()
    return stripped == "" or stripped in MISSING_TOKENS


def parse_number(raw):
    """Parse a number such as '42', '3.14' or '$1,234.56'.

    Returns (value, is_integer) or None if the text is not a finite number.
    """
    text = raw.strip().translate(CURRENCY_TABLE)
    if THOUSANDS_RE.match(text):
        text = text.replace(",", "")
    if not NUMBER_RE.match(text):
        return None
    value = float(text)
    if not math.isfinite(value):
        return None
    return value, bool(INTEGER_RE.match(text))


def parse_date(raw):
    """Parse an ISO date (YYYY-MM-DD). Returns a datetime.date or None."""
    text = raw.strip()
    if not DATE_RE.match(text):
        return None
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


def _safe_literal_eval(text):
    """ast.literal_eval that returns None instead of raising on bad input."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def parse_range(raw):
    """Parse "{'lower_bound': '100', 'upper_bound': '199'}".

    Returns (lower, upper) as floats, where upper is None for open-ended bins,
    or None if the text is not a valid range.
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    obj = _safe_literal_eval(text)
    if not isinstance(obj, dict) or "lower_bound" not in obj:
        return None
    lower = parse_number(str(obj["lower_bound"]))
    if lower is None:
        return None
    upper = None
    if "upper_bound" in obj:
        parsed_upper = parse_number(str(obj["upper_bound"]))
        if parsed_upper is None:
            return None
        upper = parsed_upper[0]
    return lower[0], upper


def parse_list(raw):
    """Parse "['a', 'b']". Returns a list of strings or None."""
    text = raw.strip()
    if not text.startswith("["):
        return None
    obj = _safe_literal_eval(text)
    if not isinstance(obj, list):
        return None
    return [str(item) for item in obj]


def classify_value(raw):
    """Classify one non-missing cell. Returns (kind, parsed_payload)."""
    number = parse_number(raw)
    if number is not None:
        return "number", number
    date = parse_date(raw)
    if date is not None:
        return "date", date
    value_range = parse_range(raw)
    if value_range is not None:
        return "range", value_range
    items = parse_list(raw)
    if items is not None:
        return "list", items
    return "text", None


# --------------------------------------------------------------------------
# Statistics (written by hand on purpose)
# --------------------------------------------------------------------------

def compute_numeric_stats(values):
    """Count, mean, min, max, sample std dev and median of a list of numbers.

    Safe on empty input and on a single value (undefined stats are None).
    """
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": None, "min": None, "max": None,
                "std": None, "median": None}

    mean = math.fsum(values) / n            # fsum avoids accumulated rounding error
    if n >= 2:
        variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = None

    ordered = sorted(values)
    middle = n // 2
    if n % 2 == 1:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2

    return {"count": n, "mean": mean, "min": ordered[0], "max": ordered[-1],
            "std": std, "median": median}


def compute_categorical_stats(counter, top_n=DEFAULT_TOP_N):
    """Count, unique values, mode and top-N for a Counter of non-missing values."""
    total = sum(counter.values())
    if total == 0:
        return {"count": 0, "unique": 0, "mode": None, "mode_freq": 0, "top": []}
    # Sort by frequency (desc), then value (asc) so ties are deterministic.
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return {"count": total, "unique": len(counter), "mode": ranked[0][0],
            "mode_freq": ranked[0][1], "top": ranked[:top_n]}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_columns(path):
    """Stream the CSV once, building a value Counter and missing count per column.

    Storing a Counter (not every raw cell) keeps memory small on large files,
    since most columns have few distinct values. Rows that are too short are
    padded with missing cells and rows that are too long are truncated; both
    cases are counted and reported rather than crashing.
    """
    csv.field_size_limit(10 ** 8)
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            raise SystemExit("error: the CSV file is empty")

        n_cols = len(header)
        counters = [Counter() for _ in range(n_cols)]
        missing = [0] * n_cols
        n_rows = short_rows = long_rows = blank_lines = 0

        for row in reader:
            if not row:               # csv.reader yields [] for a blank line
                blank_lines += 1
                continue
            n_rows += 1
            if len(row) < n_cols:
                short_rows += 1
                row = row + [""] * (n_cols - len(row))
            elif len(row) > n_cols:
                long_rows += 1
                row = row[:n_cols]
            for i, cell in enumerate(row):
                if is_missing(cell):
                    missing[i] += 1
                else:
                    counters[i][cell] += 1

    return {"header": header, "counters": counters, "missing": missing,
            "n_rows": n_rows, "short_rows": short_rows,
            "long_rows": long_rows, "blank_lines": blank_lines}


# --------------------------------------------------------------------------
# Per-column analysis
# --------------------------------------------------------------------------

def infer_column_kind(kind_counts):
    """Pick a column's type from {kind: weighted count}. Returns (kind, mixed)."""
    total = sum(kind_counts.values())
    if total == 0:
        return "empty", False
    # sorted() first makes tie-breaking deterministic.
    kind, count = max(sorted(kind_counts.items()), key=lambda item: item[1])
    if count / total >= TYPE_THRESHOLD:
        return kind, False
    return "text", True


def analyze_column(name, counter, missing, n_rows, top_n):
    """Return a dict of every statistic for one column."""
    non_null = sum(counter.values())
    info = {
        "name": name,
        "non_null": non_null,
        "missing": missing,
        "pct_missing": (100.0 * missing / n_rows) if n_rows else 0.0,
    }

    classified = {raw: classify_value(raw) for raw in counter}
    kind_counts = Counter()
    for raw, (kind, _) in classified.items():
        kind_counts[kind] += counter[raw]
    kind, mixed = infer_column_kind(kind_counts)

    info["kind"] = kind
    info["mixed"] = mixed
    info["kind_shares"] = {k: v / non_null for k, v in sorted(kind_counts.items())}

    # Report entries that do not fit the winning type instead of hiding them.
    if kind in {"number", "date", "range", "list"}:
        misfits = [(raw, counter[raw]) for raw, (k, _) in classified.items()
                   if k != kind]
        misfits.sort(key=lambda item: (-item[1], item[0]))
        info["unparseable"] = {"count": sum(c for _, c in misfits),
                               "examples": misfits[:3]}

    # ---- numeric columns -------------------------------------------------
    if kind == "number":
        values, all_int = [], True
        for raw, (k, payload) in classified.items():
            if k == "number":
                value, is_int = payload
                all_int = all_int and is_int
                values.extend([value] * counter[raw])
        if set(values) == {0.0, 1.0}:
            info["subtype"] = "binary flag (0/1)"
        else:
            info["subtype"] = "integer" if all_int else "float"
        info["numeric"] = compute_numeric_stats(values)

    # ---- range columns: numeric summary of the derived bounds ------------
    elif kind == "range":
        lowers, uppers, mids, open_ended = [], [], [], 0
        for raw, (k, payload) in classified.items():
            if k != "range":
                continue
            lower, upper = payload
            c = counter[raw]
            lowers.extend([lower] * c)
            if upper is None:
                open_ended += c
            else:
                uppers.extend([upper] * c)
                mids.extend([(lower + upper) / 2] * c)
        info["subtype"] = "range / binned (lower_bound, upper_bound)"
        info["range_series"] = {
            "lower_bound": compute_numeric_stats(lowers),
            "upper_bound": compute_numeric_stats(uppers),
            "midpoint": compute_numeric_stats(mids),
        }
        info["open_ended"] = open_ended

    # ---- dates: earliest / latest ----------------------------------------
    elif kind == "date":
        dates = [payload for k, payload in classified.values() if k == "date"]
        if dates:
            info["earliest"] = min(dates).isoformat()
            info["latest"] = max(dates).isoformat()
            info["span_days"] = (max(dates) - min(dates)).days
        info["subtype"] = "date (YYYY-MM-DD)"

    # ---- list columns: element-level frequencies -------------------------
    elif kind == "list":
        elements, empty_lists = Counter(), 0
        for raw, (k, payload) in classified.items():
            if k != "list":
                continue
            if not payload:
                empty_lists += counter[raw]
            for item in payload:
                elements[item] += counter[raw]
        info["subtype"] = "list (stringified Python list)"
        info["empty_lists"] = empty_lists
        info["elements"] = compute_categorical_stats(elements, top_n)

    elif kind == "text":
        info["subtype"] = "text (mixed types)" if mixed else "text"
    else:
        info["subtype"] = "empty (no non-missing values)"

    # Categorical stats for every non-numeric column (spec requirement).
    if kind != "number" and kind != "empty":
        info["categorical"] = compute_categorical_stats(counter, top_n)

    return info


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def fmt(x):
    """Format a number for display; None becomes 'n/a'."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and x == int(x) and abs(x) < 1e15:
        return f"{int(x):,}"
    if isinstance(x, float):
        return f"{x:,.4f}"
    return f"{x:,}"


def shorten(text, width=DISPLAY_WIDTH):
    return text if len(text) <= width else text[: width - 3] + "..."


def print_numeric_block(stats, indent="    "):
    for key in ("count", "mean", "min", "max", "std", "median"):
        label = "std dev" if key == "std" else key
        print(f"{indent}{label:<9}{fmt(stats[key])}")


def print_categorical_block(cat, indent="    "):
    print(f"{indent}count     {fmt(cat['count'])}")
    print(f"{indent}unique    {fmt(cat['unique'])}")
    if cat["mode"] is not None:
        print(f"{indent}mode      {shorten(cat['mode'])}  "
              f"(frequency {fmt(cat['mode_freq'])})")
    print(f"{indent}top {len(cat['top'])}:")
    for value, freq in cat["top"]:
        share = 100.0 * freq / cat["count"]
        print(f"{indent}  {freq:>10,}  {share:5.1f}%  {shorten(value)}")


def print_column(index, info):
    print("-" * 78)
    print(f"[{index}] {info['name']}")
    print(f"  type: {info['subtype']}")
    print(f"  non-missing: {fmt(info['non_null'])}   "
          f"missing: {fmt(info['missing'])} ({info['pct_missing']:.2f}%)")

    if info.get("mixed"):
        shares = ", ".join(f"{k} {100 * v:.1f}%" for k, v in info["kind_shares"].items())
        print(f"  note: no single type reaches {TYPE_THRESHOLD:.0%}; "
              f"treated as text ({shares})")

    unparseable = info.get("unparseable")
    if unparseable and unparseable["count"]:
        examples = "; ".join(f"{shorten(v, 30)!r} x{c:,}" for v, c in unparseable["examples"])
        print(f"  note: {fmt(unparseable['count'])} value(s) did not fit the "
              f"{info['kind']} type and were excluded. Examples: {examples}")

    if "numeric" in info:
        print("  numeric statistics:")
        print_numeric_block(info["numeric"])

    if "range_series" in info:
        print(f"  open-ended bins (no upper bound): {fmt(info['open_ended'])}")
        for series, stats in info["range_series"].items():
            print(f"  numeric statistics on {series}:")
            print_numeric_block(stats)

    if "earliest" in info:
        print(f"  earliest: {info['earliest']}   latest: {info['latest']}   "
              f"span: {fmt(info['span_days'])} days")

    if "categorical" in info:
        label = "categorical statistics (raw values)" if info["kind"] in {"range", "list"} \
            else "categorical statistics"
        print(f"  {label}:")
        print_categorical_block(info["categorical"])

    if "elements" in info:
        print(f"  empty lists: {fmt(info['empty_lists'])}")
        print("  individual list elements:")
        print_categorical_block(info["elements"])


def print_report(path, loaded, columns):
    n_rows, header = loaded["n_rows"], loaded["header"]
    print("=" * 78)
    print("DATASET OVERVIEW")
    print("=" * 78)
    print(f"file:            {path}")
    print(f"size:            {os.path.getsize(path) / 1_000_000:,.1f} MB")
    print(f"total rows:      {n_rows:,}   (header row excluded)")
    print(f"total columns:   {len(header)}")
    if loaded["short_rows"] or loaded["long_rows"] or loaded["blank_lines"]:
        print(f"structural issues: {loaded['short_rows']:,} short row(s) padded, "
              f"{loaded['long_rows']:,} long row(s) truncated, "
              f"{loaded['blank_lines']:,} blank line(s) skipped")
    else:
        print("structural issues: none (every row has the same number of fields)")
    print(f"type summary:    "
          + ", ".join(f"{n} {k}" for k, n in
                      sorted(Counter(c["kind"] for c in columns).items())))
    print()

    print("COLUMN SCHEMA: inferred type and missing values")
    print(f"{'#':>3}  {'column':<44}{'type':<12}{'missing':>9}{'%':>8}")
    for i, info in enumerate(columns):
        kind_label = info["kind"] if not info["mixed"] else "text*"
        print(f"{i:>3}  {shorten(info['name'], 43):<44}{kind_label:<12}"
              f"{info['missing']:>9,}{info['pct_missing']:>7.2f}%")
    if any(c["mixed"] for c in columns):
        print("  * mixed-type column treated as text")
    print()
    print("Standard deviation is the sample std dev (n - 1 denominator).")
    print()

    print("=" * 78)
    print("COLUMN DETAIL")
    print("=" * 78)
    for i, info in enumerate(columns):
        print_column(i, info)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Descriptive statistics for a CSV using only the standard library.")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV,
                        help=f"path to the CSV file (default: {DEFAULT_CSV})")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help="how many top values to list (default: 5)")
    parser.add_argument("--json-out", metavar="PATH",
                        help="also write all results to this JSON file "
                             "(handy for comparing against the Pandas script)")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not os.path.isfile(args.csv_path):
        sys.exit(f"error: file not found: {args.csv_path}\n"
                 "Download the dataset and place it there, or pass its path "
                 "as the first argument.")

    loaded = load_columns(args.csv_path)
    columns = [
        analyze_column(name, counter, missing, loaded["n_rows"], args.top)
        for name, counter, missing in
        zip(loaded["header"], loaded["counters"], loaded["missing"])
    ]
    print_report(args.csv_path, loaded, columns)

    if args.json_out:
        payload = {
            "file": os.path.basename(args.csv_path),
            "rows": loaded["n_rows"],
            "columns": columns,
        }
        with open(args.json_out, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, default=str)
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
