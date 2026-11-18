"""Export modeling data + model risk score into a static JSON for the React frontend.

Usage (from anywhere):
    python data_science/scripts/export_frontend_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import SIGNAL_RULES, add_priority_score  # noqa: E402

CUTOFF = pd.Timestamp("2025-03-31")
WEEKS = 6
MODEL_PATH = ROOT / "model_artifacts" / "best_model.joblib"
# Tier cut-offs on the model probability. 0.50 is the model's own decision
# threshold; 0.75 separates clear risk from the middle band.
ACTIVE_BELOW = 0.50
AT_RISK_ABOVE = 0.75


def risk_tier(probability: float) -> str:
    if probability >= AT_RISK_ABOVE:
        return "At risk"
    if probability >= ACTIVE_BELOW:
        return "Need attention"
    return "Active"


def weekly_counts(sessions: pd.DataFrame) -> pd.DataFrame:
    """Attended/scheduled count per customer for the last WEEKS weeks before cutoff."""
    in_window = sessions[sessions["scheduled_at"] <= CUTOFF].copy()
    week = ((CUTOFF - in_window["scheduled_at"]).dt.days // 7)
    in_window = in_window[week < WEEKS].assign(week=week[week < WEEKS])
    attended = (
        in_window[in_window["status"] == "attended"]
        .groupby(["customer_id", "week"]).size().unstack(fill_value=0)
        .reindex(columns=range(WEEKS), fill_value=0)
    )
    scheduled = in_window.groupby(["customer_id", "week"]).size().unstack(fill_value=0) \
        .reindex(columns=range(WEEKS), fill_value=0)
    # week 0 = most recent; reverse so the arrays read oldest -> newest
    combined = pd.concat(
        {"attended": attended[list(range(WEEKS))[::-1]], "scheduled": scheduled[list(range(WEEKS))[::-1]]},
        axis=1,
    )
    return combined.fillna(0)


def cohort_weekly(sessions: pd.DataFrame, reschedules: pd.DataFrame) -> dict:
    """Whole-cohort attendance rate and reschedule count per week (oldest first)."""
    in_window = sessions[sessions["scheduled_at"] <= CUTOFF].copy()
    week = (CUTOFF - in_window["scheduled_at"]).dt.days // 7
    in_window = in_window[week < WEEKS].assign(week=week[week < WEEKS])
    total = in_window.groupby("week").size().reindex(range(WEEKS), fill_value=0)
    attended = in_window[in_window["status"] == "attended"].groupby("week").size() \
        .reindex(range(WEEKS), fill_value=0)
    resc = reschedules[reschedules["requested_at"] <= CUTOFF].copy()
    resc_week = (CUTOFF - resc["requested_at"]).dt.days // 7
    resc_count = resc[resc_week < WEEKS].assign(week=resc_week[resc_week < WEEKS]) \
        .groupby("week").size().reindex(range(WEEKS), fill_value=0)

    order = list(range(WEEKS))[::-1]
    labels = [f"{(CUTOFF - pd.Timedelta(days=7 * (w + 1) - 6)).strftime('%b %d')}" for w in order]
    return {
        "labels": labels,
        "attendance": [round(float(attended[w] / total[w] * 100), 1) if total[w] else 0.0 for w in order],
        "reschedules": [int(resc_count[w]) for w in order],
    }


def build_customers(modeling: pd.DataFrame, weekly: pd.DataFrame, customers: pd.DataFrame, artifact: dict) -> list[dict]:
    scored = add_priority_score(modeling)
    scored = scored.merge(
        customers[["customer_id", "coach_id", "has_personal_training", "preferred_visits_per_week"]],
        on="customer_id", how="left",
    )
    flags = {key: predicate(scored[column]).to_numpy() for key, _, column, predicate in SIGNAL_RULES}
    # Score each customer with the trained model, not the rule count.
    probabilities = artifact["model"].predict_proba(scored[artifact["feature_columns"]])[:, 1]

    rows = []
    for position, row in enumerate(scored.itertuples(index=False)):
        cid = row.customer_id
        series = weekly.loc[cid] if cid in weekly.index else None
        probability = round(float(probabilities[position]), 3)
        signals = [
            {"key": key, "label": label, "value": round(float(getattr(row, column)), 3)}
            for key, label, column, _ in SIGNAL_RULES if flags[key][position]
        ]
        rows.append({
            "id": cid,
            "packageType": row.package_type,
            "coachId": row.coach_id,
            "personalTraining": bool(row.has_personal_training),
            "preferredVisitsPerWeek": int(row.preferred_visits_per_week),
            "attendanceRate": round(float(row.attendance_rate), 3),
            "recentAttendanceRate": round(float(row.recent_attendance_rate), 3),
            "attendanceTrend": round(float(row.attendance_trend), 3),
            "attendedSessions": int(row.attended_sessions),
            "missedSessions": int(row.missed_sessions),
            "visitFrequency": round(float(row.visit_frequency), 3),
            "maxConsecutiveAbsences": int(row.max_consecutive_absences),
            "daysSinceLastVisit": int(row.days_since_last_visit),
            "packageUsageRatio": round(float(row.package_usage_ratio), 3),
            "daysToPackageExpiry": int(row.days_to_package_expiry),
            "renewalCount": int(row.renewal_count),
            "rescheduleCount": int(row.reschedule_count),
            "rescheduleRate": round(float(row.reschedule_rate), 3),
            "adminContactCount": int(row.admin_contact_count),
            "noResponseCount": int(row.no_response_count),
            "probability": probability,
            "risk": risk_tier(probability),
            "ruleScore": int(row.priority_score),
            "signals": signals,
            "weekly": {
                "attended": [int(v) for v in series["attended"]] if series is not None else [0] * WEEKS,
                "scheduled": [int(v) for v in series["scheduled"]] if series is not None else [0] * WEEKS,
            },
        })
    return rows


def main() -> None:
    modeling = pd.read_csv(ROOT / "data_ready" / "modeling_df.csv")
    sessions = pd.read_csv(ROOT / "data_dummy" / "sessions.csv", parse_dates=["scheduled_at"])
    reschedules = pd.read_csv(ROOT / "data_dummy" / "reschedules.csv", parse_dates=["requested_at"])
    customers = pd.read_csv(ROOT / "data_dummy" / "customers.csv")
    artifact = joblib.load(MODEL_PATH)

    # needs_follow_up is the future outcome label; it is deliberately not exported.
    modeling = modeling.drop(columns=["needs_follow_up"])
    rows = build_customers(modeling, weekly_counts(sessions), customers, artifact)

    payload = {
        "meta": {
            "cutoff": CUTOFF.strftime("%Y-%m-%d"),
            "horizon": "2025-04-01..2025-04-27",
            "totalCustomers": len(rows),
            "scoring": "Random Forest model probability, not a rule count",
            "modelName": artifact.get("model_name", "Random Forest"),
            "modelMetrics": artifact.get("validation_metrics", {}),
            "thresholds": {"activeBelow": ACTIVE_BELOW, "atRiskAbove": AT_RISK_ABOVE},
            "weekly": cohort_weekly(sessions, reschedules),
        },
        "customers": rows,
    }

    assert len(rows) == 3000, f"expected 3000 customers, got {len(rows)}"
    assert all(row["signals"] is not None and row["coachId"] for row in rows), "null fields in export"

    output = ROOT.parent / "frontend" / "public" / "data" / "customers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    counts = pd.Series([row["risk"] for row in rows]).value_counts()
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    print(counts.to_string())


if __name__ == "__main__":
    main()
