#!/usr/bin/env python3
"""
Filter and sort stock CSV data using YAML filters exported from stock_search.
"""

from __future__ import annotations

import argparse
import locale
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


NUMERIC_COLUMNS = {
    "時価総額",
    "PBR",
    "売上高",
    "営業利益",
    "営業利益率",
    "当期純利益",
    "純利益率",
    "ROE",
    "自己資本比率",
    "PER(会予)",
    "PER(過去12ヶ月)",
    "PER(前年度)",
    "配当方向性",
    "配当利回り",
    "EPS(過去12ヶ月)",
    "EPS(予想)",
    "EPS(前年度)",
    "負債",
    "流動負債",
    "流動資産",
    "総負債",
    "現金及び現金同等物",
    "投資有価証券",
    "ネットキャッシュ",
    "ネットキャッシュ（流動資産-負債）",
    "ネットキャッシュ比率",
}


NUMERIC_FILTERS: Dict[str, Tuple[Sequence[str], str, float]] = {
    "marketCapMin": (["時価総額"], "min", 1_000_000),
    "marketCapMax": (["時価総額"], "max", 1_000_000),
    "pbrMin": (["PBR"], "min", 1),
    "pbrMax": (["PBR"], "max", 1),
    "roeMin": (["ROE"], "min", 0.01),
    "roeMax": (["ROE"], "max", 0.01),
    "revenueMin": (["売上高"], "min", 1_000_000),
    "revenueMax": (["売上高"], "max", 1_000_000),
    "operatingProfitMin": (["営業利益"], "min", 1_000_000),
    "operatingProfitMax": (["営業利益"], "max", 1_000_000),
    "operatingMarginMin": (["営業利益率"], "min", 0.01),
    "operatingMarginMax": (["営業利益率"], "max", 0.01),
    "netProfitMin": (["当期純利益"], "min", 1_000_000),
    "netProfitMax": (["当期純利益"], "max", 1_000_000),
    "netMarginMin": (["純利益率"], "min", 0.01),
    "netMarginMax": (["純利益率"], "max", 0.01),
    "equityRatioMin": (["自己資本比率"], "min", 0.01),
    "equityRatioMax": (["自己資本比率"], "max", 0.01),
    "forwardPEMin": (["PER(会予)"], "min", 1),
    "forwardPEMax": (["PER(会予)"], "max", 1),
    "trailingPEMin": (["PER(過去12ヶ月)"], "min", 1),
    "trailingPEMax": (["PER(過去12ヶ月)"], "max", 1),
    "previousYearPEMin": (["PER(前年度)"], "min", 1),
    "previousYearPEMax": (["PER(前年度)"], "max", 1),
    "dividendDirectionMin": (["配当方向性"], "min", 0.01),
    "dividendDirectionMax": (["配当方向性"], "max", 0.01),
    "dividendYieldMin": (["配当利回り"], "min", 0.01),
    "dividendYieldMax": (["配当利回り"], "max", 0.01),
    "trailingEpsMin": (["EPS(過去12ヶ月)"], "min", 1),
    "trailingEpsMax": (["EPS(過去12ヶ月)"], "max", 1),
    "forwardEpsMin": (["EPS(予想)"], "min", 1),
    "forwardEpsMax": (["EPS(予想)"], "max", 1),
    "previousYearEpsMin": (["EPS(前年度)"], "min", 1),
    "previousYearEpsMax": (["EPS(前年度)"], "max", 1),
    "totalLiabilitiesMin": (["負債"], "min", 1_000_000),
    "totalLiabilitiesMax": (["負債"], "max", 1_000_000),
    "currentLiabilitiesMin": (["流動負債"], "min", 1_000_000),
    "currentLiabilitiesMax": (["流動負債"], "max", 1_000_000),
    "currentAssetsMin": (["流動資産"], "min", 1_000_000),
    "currentAssetsMax": (["流動資産"], "max", 1_000_000),
    "totalDebtMin": (["総負債"], "min", 1_000_000),
    "totalDebtMax": (["総負債"], "max", 1_000_000),
    "cashMin": (["現金及び現金同等物"], "min", 1_000_000),
    "cashMax": (["現金及び現金同等物"], "max", 1_000_000),
    "investmentsMin": (["投資有価証券"], "min", 1_000_000),
    "investmentsMax": (["投資有価証券"], "max", 1_000_000),
    "netCashMin": (
        ["ネットキャッシュ", "ネットキャッシュ（流動資産-負債）"],
        "min",
        1_000_000,
    ),
    "netCashMax": (
        ["ネットキャッシュ", "ネットキャッシュ（流動資産-負債）"],
        "max",
        1_000_000,
    ),
    "netCashRatioMin": (["ネットキャッシュ比率"], "min", 0.01),
    "netCashRatioMax": (["ネットキャッシュ比率"], "max", 0.01),
}

SORT_ALIASES: Dict[str, Sequence[str]] = {
    "ネットキャッシュ": ["ネットキャッシュ", "ネットキャッシュ（流動資産-負債）"],
    "ネットキャッシュ（流動資産-負債）": [
        "ネットキャッシュ（流動資産-負債）",
        "ネットキャッシュ",
    ],
    "銘柄コード": ["銘柄コード", "コード"],
    "コード": ["コード", "銘柄コード"],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and sort stock CSV data using YAML filters."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input CSV file path (UTF-8, comma-separated).",
    )
    parser.add_argument(
        "--filters",
        "-f",
        required=True,
        help="YAML file path exported from stock_search.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV file path.",
    )
    return parser.parse_args()


def _unescape_double_quoted(value: str) -> str:
    result = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            mapping = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
            result.append(mapping.get(nxt, nxt))
            i += 2
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _parse_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return _unescape_double_quoted(value[1:-1])
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_yaml_minimal(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_list_key: Optional[str] = None

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if line.startswith("filters:"):
            result["filters"] = {}
            current_section = "filters"
            current_list_key = None
            if line.strip() == "filters: {}":
                continue
            continue
        if line.startswith("sort:"):
            result["sort"] = {}
            current_section = "sort"
            current_list_key = None
            continue
        if current_section is None:
            continue
        if line.startswith("    - "):
            if current_list_key is None:
                continue
            value = _parse_yaml_scalar(line[len("    - ") :].strip())
            section = result.setdefault(current_section, {})
            section.setdefault(current_list_key, []).append(value)
            continue
        if line.startswith("  "):
            key, rest = line.strip().split(":", 1)
            value = rest.strip()
            section = result.setdefault(current_section, {})
            if value == "":
                section[key] = []
                current_list_key = key
            else:
                section[key] = _parse_yaml_scalar(value)
                current_list_key = None
            continue

    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_yaml_minimal(text)

    try:
        data = yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse YAML: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("倍", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace("円", "", regex=False)
    cleaned = cleaned.str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def resolve_column(candidates: Sequence[str], available: Iterable[str]) -> Optional[str]:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    return None


def resolve_sort_column(sort_key: str, available: Iterable[str]) -> Optional[str]:
    candidates = SORT_ALIASES.get(sort_key, [sort_key])
    return resolve_column(candidates, available)


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value).strip()]


def parse_filter_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def build_filter_mask(
    df_raw: pd.DataFrame,
    df_numeric: pd.DataFrame,
    filters: Dict[str, Any],
) -> pd.Series:
    mask = pd.Series(True, index=df_raw.index)

    company_name = filters.get("companyName")
    if isinstance(company_name, str) and company_name.strip():
        if "会社名" in df_raw.columns:
            series = df_raw["会社名"].fillna("")
            mask &= series.str.contains(
                company_name, case=False, regex=False, na=False
            )

    stock_code = filters.get("stockCode")
    if isinstance(stock_code, str) and stock_code.strip():
        code_column = resolve_column(["銘柄コード", "コード"], df_raw.columns)
        if code_column:
            series = df_raw[code_column].fillna("")
            mask &= series.astype(str).str.contains(
                stock_code, regex=False, na=False
            )

    industries = normalize_list(filters.get("industries"))
    if industries and "業種" in df_raw.columns:
        mask &= df_raw["業種"].isin(industries)

    market = normalize_list(filters.get("market"))
    if market and "優先市場" in df_raw.columns:
        mask &= df_raw["優先市場"].isin(market)

    prefecture = normalize_list(filters.get("prefecture"))
    if prefecture and "都道府県" in df_raw.columns:
        mask &= df_raw["都道府県"].isin(prefecture)

    for key, (columns, bound, scale) in NUMERIC_FILTERS.items():
        filter_value = parse_filter_number(filters.get(key))
        if filter_value is None:
            continue
        column = resolve_column(columns, df_numeric.columns)
        if not column:
            continue
        threshold = filter_value * scale
        series = df_numeric[column]
        if bound == "min":
            mask &= series.isna() | (series >= threshold)
        else:
            mask &= series.isna() | (series <= threshold)

    return mask


def _init_locale() -> None:
    try:
        locale.setlocale(locale.LC_COLLATE, "ja_JP.UTF-8")
    except locale.Error:
        pass


def sort_dataframe(
    df_raw: pd.DataFrame,
    df_numeric: pd.DataFrame,
    sort_config: Dict[str, Any],
) -> pd.DataFrame:
    sort_key = sort_config.get("key")
    direction = sort_config.get("direction", "asc")
    if not isinstance(sort_key, str) or not sort_key.strip():
        return df_raw

    direction = str(direction).lower()
    if direction not in ("asc", "desc"):
        return df_raw

    column = resolve_sort_column(sort_key, df_raw.columns)
    if not column:
        return df_raw

    is_numeric = column in NUMERIC_COLUMNS
    series = df_numeric[column] if is_numeric else df_raw[column]
    collate = locale.strxfrm
    reverse = direction == "desc"

    def sort_key_func(value: Any) -> Tuple[bool, Any]:
        if is_numeric:
            if pd.isna(value):
                return (True, 0.0)
            return (False, float(value))
        if value is None or value == "":
            return (True, "")
        return (False, collate(str(value)))

    order = sorted(
        series.index,
        key=lambda idx: sort_key_func(series.at[idx]),
        reverse=reverse,
    )
    return df_raw.loc[order]


def load_csv_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    df_numeric = df_raw.copy()
    for column in NUMERIC_COLUMNS:
        if column in df_numeric.columns:
            df_numeric[column] = clean_numeric_series(df_numeric[column])
    return df_raw, df_numeric


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    yaml_path = Path(args.filters)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input CSV not found: %s", input_path)
        return 1
    if not yaml_path.exists():
        logger.error("YAML file not found: %s", yaml_path)
        return 1

    try:
        config = load_yaml(yaml_path)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    filters = config.get("filters")
    if filters is None:
        filters = {}
    if not isinstance(filters, dict):
        logger.error("filters must be a mapping in YAML.")
        return 1

    sort_config = config.get("sort")
    if sort_config is not None and not isinstance(sort_config, dict):
        logger.warning("sort is not a mapping, skipping sort.")
        sort_config = None

    df_raw, df_numeric = load_csv_data(input_path)
    mask = build_filter_mask(df_raw, df_numeric, filters)
    filtered_raw = df_raw[mask].copy()
    filtered_numeric = df_numeric[mask].copy()

    _init_locale()
    if sort_config:
        filtered_raw = sort_dataframe(filtered_raw, filtered_numeric, sort_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_raw.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Input rows: %d", len(df_raw))
    logger.info("Filtered rows: %d", len(filtered_raw))
    logger.info("Output saved: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
