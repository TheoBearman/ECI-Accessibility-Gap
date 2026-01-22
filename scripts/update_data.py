
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data source
ECI_SCORES_URL = "https://epoch.ai/data/eci_scores.csv"

def get_rank(
    df: pd.DataFrame,
    n: int | None = None,
    sort_col: str = "date",
    val_col: str = "eci",
) -> pd.Series:
    """
    Cumulative rank of *val_col* up to each row, ordered by *sort_col*.
    """
    ordered = df.sort_values(sort_col, kind="mergesort", na_position="last").reset_index()
    vals = ordered[val_col]
    ranks = pd.Series(np.nan, index=ordered.index, dtype=float)

    seen = []
    for idx, v in enumerate(vals):
        if pd.isna(v):
            continue
        rank = 1 + sum(prev > v for prev in seen)
        ranks.iloc[idx] = rank
        seen.append(v)

    if n is not None:
        ranks = ranks.where(ranks <= n)

    ranks.index = ordered["index"]
    return ranks.reindex(df.index)

def fetch_eci_data() -> pd.DataFrame:
    """Fetch ECI scores from Epoch AI."""
    logger.info("Fetching data from Epoch AI...")
    try:
        df = pd.read_csv(ECI_SCORES_URL)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Derive eci_std from confidence intervals (assuming 90% CI)
        # For 90% CI, z = 1.645, so std = (ci_high - ci_low) / (2 * 1.645)
        if "eci_ci_low" in df.columns and "eci_ci_high" in df.columns:
            df["eci_std"] = (df["eci_ci_high"] - df["eci_ci_low"]) / (2 * 1.645)
        else:
            df["eci_std"] = pd.NA
        
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

def calculate_horizontal_gaps(df: pd.DataFrame) -> list[dict]:
    """Calculate horizontal gaps."""
    if len(df) == 0:
        return []

    df_open = df[df["Open"]].sort_values("date")
    df_closed = df[~df["Open"]].sort_values("date")

    gaps = []

    for _, closed_row in df_closed.iterrows():
        closed_eci = closed_row["eci"]
        closed_date = closed_row["date"]

        if pd.isna(closed_eci) or pd.isna(closed_date):
            continue

        matching_open = None
        match_type = None
        for _, open_row in df_open.iterrows():
            if pd.isna(open_row["eci"]) or pd.isna(open_row["date"]):
                continue

            if open_row["date"] <= closed_date:
                continue

            # Match if open model is within 1 ECI point of closed model
            if open_row["eci"] >= closed_eci - 1:
                matching_open = open_row
                match_type = "exact"
                break

        if matching_open is not None:
            gap_days = (matching_open["date"] - closed_date).days
            gap_months = gap_days / 30.5

            gaps.append({
                "closed_model": closed_row.get("Model", "Unknown"),
                "closed_date": closed_date.isoformat(),
                "closed_eci": float(closed_eci),
                "open_model": matching_open.get("Model", "Unknown"),
                "open_date": matching_open["date"].isoformat(),
                "open_eci": float(matching_open["eci"]),
                "gap_months": round(gap_months, 1),
                "matched": True,
                "match_type": match_type,
            })
        else:
            now = datetime.now()
            gap_days = (now - closed_date.to_pydatetime().replace(tzinfo=None)).days
            gap_months = gap_days / 30.5

            gaps.append({
                "closed_model": closed_row.get("Model", "Unknown"),
                "closed_date": closed_date.isoformat(),
                "closed_eci": float(closed_eci),
                "open_model": None,
                "open_date": None,
                "open_eci": None,
                "gap_months": round(gap_months, 1),
                "matched": False,
            })

    return gaps

def calculate_trends(df: pd.DataFrame) -> dict:
    """Calculate trends for models before and after Apr 2024."""
    trends = {}
    cutoff = pd.Timestamp("2024-04-01")
    
    def get_stats(sub_df, name):
        if len(sub_df) < 2:
            return None
            
        dates_ordinal = sub_df["date"].map(datetime.toordinal).values
        ecis = sub_df["eci"].values
        
        lin_slope, lin_intercept, _, _, _ = linregress(dates_ordinal, ecis)
        yearly_absolute_growth = lin_slope * 365.25
        
        start_date_ord = dates_ordinal.min()
        end_date_ord = dates_ordinal.max()
        
        lin_start_eci = lin_slope * start_date_ord + lin_intercept
        lin_end_eci = lin_slope * end_date_ord + lin_intercept
        
        valid_indices = ecis > 0
        if not np.any(valid_indices):
            return None
            
        log_ecis = np.log(ecis[valid_indices])
        log_dates = dates_ordinal[valid_indices]
        
        exp_slope, _, _, _, _ = linregress(log_dates, log_ecis)
        k_annual = exp_slope * 365.25
        
        pct_growth = (np.exp(k_annual) - 1) * 100
        multiples_per_year = np.exp(k_annual)
        doubling_time_years = np.log(2) / k_annual if k_annual > 0 else float('inf')

        return {
            "name": name,
            "absolute_growth_per_year": round(yearly_absolute_growth, 2),
            "percentage_growth_annualized": round(pct_growth, 1),
            "multiples_per_year": round(multiples_per_year, 2),
            "doubling_time_years": round(doubling_time_years, 2),
            "start_point": {
                "date": datetime.fromordinal(int(start_date_ord)).isoformat(),
                "eci": lin_start_eci
            },
            "end_point": {
                "date": datetime.fromordinal(int(end_date_ord)).isoformat(),
                "eci": lin_end_eci
            }
        }

    pre_2024 = df[df["date"] < cutoff]
    trends["pre_apr_2024"] = get_stats(pre_2024, "Pre-Apr 2024")
    
    post_2024 = df[df["date"] >= cutoff]
    trends["post_apr_2024"] = get_stats(post_2024, "Post-Apr 2024")
    
    return trends

def estimate_current_gap(gaps: list[dict], matched_gaps_months: list[float]) -> dict:
    """
    Estimate the current gap using unmatched models as censored observations.

    Uses a simple Bayesian-inspired approach: unmatched models provide lower bounds
    on the current gap. We weight by how long each model has been unmatched.
    """
    unmatched = [g for g in gaps if not g["matched"]]

    if not unmatched:
        return {
            "estimated_current_gap": 0,
            "min_current_gap": 0,
            "confidence": "high",
            "unmatched_ages": [],
        }

    # Get ages of unmatched models (minimum gap values)
    unmatched_ages = sorted([g["gap_months"] for g in unmatched], reverse=True)

    # The minimum current gap is the age of the oldest unmatched model
    min_current_gap = max(unmatched_ages) if unmatched_ages else 0

    # Estimate: Use survival analysis intuition
    # If we have matched gaps averaging X months and unmatched models Y months old,
    # the current gap is likely at least as large as the oldest unmatched model
    #
    # Simple estimate: weighted average where older unmatched models get more weight
    # Plus an adjustment factor based on historical variance

    if matched_gaps_months and len(matched_gaps_months) >= 3:
        historical_mean = np.mean(matched_gaps_months)
        historical_std = np.std(matched_gaps_months)

        # Bayesian update: prior is historical, evidence is unmatched ages
        # Simple heuristic: estimate is max of (historical mean, weighted unmatched)
        weights = np.array(unmatched_ages) / sum(unmatched_ages) if sum(unmatched_ages) > 0 else np.ones(len(unmatched_ages))
        weighted_unmatched = np.average(unmatched_ages, weights=weights)

        # Add uncertainty premium - gaps tend to be longer when there are many unmatched
        uncertainty_premium = 0.5 * len(unmatched) * (historical_std / historical_mean if historical_mean > 0 else 0.2)

        estimated = max(weighted_unmatched + uncertainty_premium, min_current_gap)
        confidence = "medium" if len(unmatched) <= 3 else "low"
    else:
        # Not enough historical data, use simple average of unmatched
        estimated = np.mean(unmatched_ages) * 1.2 if unmatched_ages else 0
        confidence = "low"

    return {
        "estimated_current_gap": round(estimated, 1),
        "min_current_gap": round(min_current_gap, 1),
        "confidence": confidence,
        "unmatched_ages": [round(a, 1) for a in unmatched_ages],
    }


def calculate_historical_gaps(df: pd.DataFrame) -> list[dict]:
    """
    Calculate the gap metric at various points in history.
    This shows how the gap has evolved over time.

    Approach: At each point in time, calculate how far behind the open frontier
    is from the closed frontier. This is measured as:
    - Time since the closed frontier first achieved its current ECI level
      minus time since the open frontier achieved its current ECI level

    Simpler approach: Find when the best open model was released, and when
    a closed model first achieved that ECI level. The gap is the time difference.
    """
    df = df.dropna(subset=["date", "eci"]).copy()
    df = df.sort_values("date")

    if len(df) < 2:
        return []

    historical_gaps = []

    # Sample monthly from first closed model to now
    min_date = df["date"].min()
    max_date = pd.Timestamp(datetime.now())

    # Generate monthly checkpoints
    current = pd.Timestamp(min_date) + pd.DateOffset(months=6)  # Start 6 months in

    while current <= max_date:
        # Get all models released before this date
        df_at_time = df[df["date"] <= current].copy()

        df_open = df_at_time[df_at_time["Open"]]
        df_closed = df_at_time[~df_at_time["Open"]]

        if len(df_open) == 0 or len(df_closed) == 0:
            current += pd.DateOffset(months=1)
            continue

        # Find the best open model at this time
        best_open = df_open.loc[df_open["eci"].idxmax()]
        best_open_eci = best_open["eci"]
        best_open_date = best_open["date"]

        # Find the best closed model at this time
        best_closed = df_closed.loc[df_closed["eci"].idxmax()]
        best_closed_eci = best_closed["eci"]

        # Find the first closed model to achieve the best open's ECI level
        closed_at_open_level = df_closed[df_closed["eci"] >= best_open_eci - 1].sort_values("date")

        if len(closed_at_open_level) > 0:
            first_closed_at_level = closed_at_open_level.iloc[0]
            # Gap is: when did closed first hit this level vs when did open hit it
            gap_days = (best_open_date - first_closed_at_level["date"]).days
            gap_months = gap_days / 30.5

            # If gap is negative, open was first (unusual but possible)
            gap_months = max(0, gap_months)

            # Determine if the frontier is "matched" (open has caught up to closed frontier)
            is_matched = best_open_eci >= best_closed_eci - 1
        else:
            # No closed model at this level yet (open is ahead - very rare)
            gap_months = 0
            is_matched = True

        historical_gaps.append({
            "date": current.isoformat(),
            "gap_months": round(float(gap_months), 1),
            "matched": bool(is_matched),
            "reference_model": best_closed.get("Model", "Unknown"),
            "reference_eci": round(float(best_closed_eci), 1),
            "open_frontier_model": best_open.get("Model", "Unknown"),
            "open_frontier_eci": round(float(best_open_eci), 1),
        })

        current += pd.DateOffset(months=1)

    return historical_gaps


def calculate_statistics(df: pd.DataFrame, gaps: list[dict]) -> dict:
    """Calculate summary statistics."""
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    if len(df_open) == 0 or len(df_closed) == 0:
        return {
            "avg_horizontal_gap_months": 0,
            "std_horizontal_gap": 0,
            "ci_90_low": 0,
            "ci_90_high": 0,
            "current_vertical_gap": 0,
            "total_matched": 0,
            "total_unmatched": len(gaps),
            "current_gap_estimate": {
                "estimated_current_gap": 0,
                "min_current_gap": 0,
                "confidence": "low",
                "unmatched_ages": [],
            },
        }

    start_eci = max(df_open["eci"].min(), df_closed["eci"].min())
    end_eci = max(df_open["eci"].max(), df_closed["eci"].max())

    horizontal_gaps = []
    df_open_possible = df_open.sort_values("date").copy()

    for cur_eci in np.linspace(start_eci, end_eci, 100):
        closed_candidates = df_closed[df_closed["eci"] >= cur_eci].sort_values("date")
        if len(closed_candidates) == 0:
            continue
        cur_closed_model = closed_candidates.iloc[0]

        cur_open_model = None
        for _, row in df_open_possible.iterrows():
            if pd.isna(row["eci"]) or pd.isna(row["date"]):
                continue
            if row["eci"] >= cur_eci:
                cur_open_model = row
                gap = (cur_open_model["date"] - cur_closed_model["date"]).days / 30.5
                horizontal_gaps.append(gap)
                break

        if cur_open_model is None:
            now = datetime.now()
            gap = (now - cur_closed_model["date"].to_pydatetime().replace(tzinfo=None)).days / 30.5
            horizontal_gaps.append(gap)

    if horizontal_gaps:
        avg_gap = np.mean(horizontal_gaps)
        std_gap = np.std(horizontal_gaps)
        ci_low = np.percentile(horizontal_gaps, 5)
        ci_high = np.percentile(horizontal_gaps, 95)
    else:
        avg_gap = std_gap = ci_low = ci_high = 0

    best_open_eci = df_open["eci"].max() if len(df_open) > 0 else 0
    best_closed_eci = df_closed["eci"].max() if len(df_closed) > 0 else 0
    vertical_gap = best_closed_eci - best_open_eci

    matched_gaps = [g for g in gaps if g["matched"]]
    matched_gaps_months = [g["gap_months"] for g in matched_gaps]

    # Estimate current gap using unmatched models
    current_gap_estimate = estimate_current_gap(gaps, matched_gaps_months)

    return {
        "avg_horizontal_gap_months": round(avg_gap, 1),
        "std_horizontal_gap": round(std_gap, 1),
        "ci_90_low": round(ci_low, 1),
        "ci_90_high": round(ci_high, 1),
        "current_vertical_gap": round(vertical_gap, 1),
        "total_matched": len(matched_gaps),
        "total_unmatched": len(gaps) - len(matched_gaps),
        "current_gap_estimate": current_gap_estimate,
    }

# Chinese tech companies/organizations
CHINA_ORGANIZATIONS = {
    "DeepSeek",
    "Alibaba",
    "Baichuan",
    "01.AI",
    "Moonshot",
    "Tsinghua",
    "Peking University",
    "ByteDance",
    "Tencent",
    "Huawei",
    "SenseTime",
    "iFlytek",
    "Zhipu AI",
}

# US tech companies/organizations
US_ORGANIZATIONS = {
    "OpenAI",
    "Anthropic",
    "Google",
    "Google DeepMind",
    "Meta",
    "Meta AI",
    "Microsoft",
    "Microsoft Research",
    "xAI",
    "NVIDIA",
    "Databricks",
    "MosaicML",
    "Salesforce",
    "Cerebras",
    "Hugging Face",
}


def is_china_org(org: str) -> bool:
    """Check if an organization is Chinese."""
    if not org:
        return False
    org_lower = org.lower()
    for china_org in CHINA_ORGANIZATIONS:
        if china_org.lower() in org_lower:
            return True
    return False


def is_us_org(org: str) -> bool:
    """Check if an organization is US-based."""
    if not org:
        return False
    org_lower = org.lower()
    for us_org in US_ORGANIZATIONS:
        if us_org.lower() in org_lower:
            return True
    return False


def process_data() -> dict[str, Any]:
    """Process ECI data and calculate gaps."""
    df = fetch_eci_data()
    df["Open"] = df["Model accessibility"].str.contains("Open", na=False)
    df["is_china"] = df["Organization"].apply(is_china_org)
    df["is_us"] = df["Organization"].apply(is_us_org)

    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    df_open["group_rank"] = get_rank(df_open, sort_col="date", val_col="eci")
    df_closed["group_rank"] = get_rank(df_closed, sort_col="date", val_col="eci")

    df_combined = pd.concat([df_open, df_closed]).sort_values("date")
    df_frontier = df_combined[df_combined["group_rank"] <= 1].copy()

    models = []
    for _, row in df_frontier.iterrows():
        models.append({
            "model": row.get("Model", row.get("model version", "Unknown")),
            "display_name": row.get("Model", row.get("Model", "Unknown")),
            "eci": float(row["eci"]) if pd.notna(row["eci"]) else None,
            "eci_std": float(row["eci_std"]) if pd.notna(row.get("eci_std")) else None,
            "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
            "organization": row.get("Organization", "Unknown"),
            "is_open": bool(row["Open"]),
            "is_china": bool(row.get("is_china", False)),
        })

    trend_models = []
    df_all_valid = df_combined.dropna(subset=["eci", "date"])
    for _, row in df_all_valid.iterrows():
        trend_models.append({
            "model": row.get("Model", row.get("model version", "Unknown")),
            "display_name": row.get("Display name", row.get("Model", "Unknown")),
            "eci": float(row["eci"]),
            "eci_std": float(row["eci_std"]) if pd.notna(row.get("eci_std")) else None,
            "date": row["date"].isoformat(),
            "organization": row.get("Organization", "Unknown"),
            "is_open": bool(row["Open"]),
            "is_china": bool(row.get("is_china", False)),
        })

    gaps = calculate_horizontal_gaps(df_frontier)
    stats = calculate_statistics(df_frontier, gaps)
    trends = calculate_trends(df_all_valid)

    # Calculate historical gaps for the timeline chart
    historical_gaps = calculate_historical_gaps(df_frontier)

    # Also calculate gaps using China vs US framing
    # Filter to only China and US models
    df_china_us = df[df["is_china"] | df["is_us"]].copy()
    df_china_us["Open"] = df_china_us["is_china"]  # China = "Open" (catching up), US = "Closed" (leading)

    df_china_models = df_china_us[df_china_us["Open"]].copy()
    df_us_models = df_china_us[~df_china_us["Open"]].copy()

    if len(df_china_models) > 0 and len(df_us_models) > 0:
        df_china_models["group_rank"] = get_rank(df_china_models, sort_col="date", val_col="eci")
        df_us_models["group_rank"] = get_rank(df_us_models, sort_col="date", val_col="eci")
        df_china_us_combined = pd.concat([df_china_models, df_us_models]).sort_values("date")
        df_china_us_frontier = df_china_us_combined[df_china_us_combined["group_rank"] <= 1].copy()

        china_gaps = calculate_horizontal_gaps(df_china_us_frontier)
        china_stats = calculate_statistics(df_china_us_frontier, china_gaps)
        china_historical = calculate_historical_gaps(df_china_us_frontier)
    else:
        china_gaps = []
        china_stats = {}
        china_historical = []

    return {
        "models": models,
        "trend_models": trend_models,
        "gaps": gaps,
        "statistics": stats,
        "trends": trends,
        "historical_gaps": historical_gaps,
        "china_framing": {
            "gaps": china_gaps,
            "statistics": china_stats,
            "historical_gaps": china_historical,
        },
        "last_updated": datetime.now().isoformat(),
    }

def main():
    try:
        data = process_data()
        with open("data.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Successfully wrote data.json")
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
