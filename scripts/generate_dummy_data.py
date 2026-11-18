"""Generate six related synthetic Gymnasyum datasets for the retention-risk prototype.

The files in data_raw/ are used only as distribution/structure references. They are
never joined to the generated tables. Every generated table uses synthetic keys and
is validated for missing values, duplicate rows, foreign-key integrity, and temporal
consistency before the script finishes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SENTINEL = "not_applicable"
NONE_REF = "NONE"
DATA_ORIGIN = "synthetic_strategy_3_reference_calibrated"

SCENARIOS = (
    "stable_active",
    "gradual_decline",
    "temporary_disruption",
    "irregular_attendance",
    "package_expiry_risk",
)
SCENARIO_PROBABILITIES = (0.45, 0.20, 0.12, 0.13, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_dummy"))
    parser.add_argument("--n-customers", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--start-date", default="2025-01-06")
    parser.add_argument("--observation-weeks", type=int, default=12)
    parser.add_argument("--horizon-weeks", type=int, default=4)
    return parser.parse_args()


def normalized_probabilities(values: Iterable[object], fallback: list[object]) -> tuple[list[object], np.ndarray]:
    series = pd.Series(list(values), dtype="object").dropna()
    if series.empty:
        return fallback, np.full(len(fallback), 1 / len(fallback))
    counts = series.astype(str).value_counts()
    labels = counts.index.tolist()
    probabilities = (counts / counts.sum()).to_numpy(dtype=float)
    return labels, probabilities


def load_references(raw_dir: Path) -> dict[str, object]:
    """Load only marginal references; no generated key is based on a raw ID."""
    attendance_path = raw_dir / "daily_gym_attendance_workout_data.csv"
    attendance = pd.read_csv(attendance_path) if attendance_path.exists() else pd.DataFrame()

    if not attendance.empty:
        age_values = pd.to_numeric(attendance.get("age"), errors="coerce").dropna()
        gender_labels, gender_probs = normalized_probabilities(
            attendance.get("gender", []), ["Female", "Male", "Other"]
        )
        package_labels, package_probs = normalized_probabilities(
            attendance.get("membership_type", []), ["Monthly", "Quarterly", "Annual"]
        )
        workout_labels, workout_probs = normalized_probabilities(
            attendance.get("workout_type", []), ["Cardio", "Strength Training", "Yoga", "HIIT"]
        )
        duration_values = pd.to_numeric(
            attendance.get("workout_duration_minutes"), errors="coerce"
        ).dropna()
        present_rate = float(
            attendance.get("attendance_status", pd.Series(dtype=str))
            .astype(str)
            .str.lower()
            .eq("present")
            .mean()
        )
    else:
        age_values = pd.Series([18, 25, 32, 40, 50, 60])
        gender_labels, gender_probs = ["Female", "Male", "Other"], np.array([0.45, 0.45, 0.10])
        package_labels, package_probs = ["Monthly", "Quarterly", "Annual"], np.array([0.50, 0.35, 0.15])
        workout_labels, workout_probs = ["Cardio", "Strength Training", "Yoga", "HIIT"], np.full(4, 0.25)
        duration_values = pd.Series([30, 45, 60, 75, 90])
        present_rate = 0.65

    reason_path = raw_dir / "reschedule_history.csv"
    if reason_path.exists():
        reason_reference = pd.read_csv(reason_path).get("reschedule_reason", pd.Series(dtype=str))
        reason_labels, reason_probs = normalized_probabilities(
            reason_reference, ["schedule_conflict", "personal", "travel", "work"]
        )
    else:
        reason_labels, reason_probs = (
            ["schedule_conflict", "personal", "travel", "work"],
            np.array([0.45, 0.25, 0.15, 0.15]),
        )

    # DadosV3 is used only as an optional reference for the age range. Its target and
    # customer IDs are deliberately not used in the generated data.
    dados_path = raw_dir / "DadosV3.csv"
    if dados_path.exists():
        try:
            dados = pd.read_csv(dados_path, sep=";", usecols=["Age"])
            dados_age_values = pd.to_numeric(dados["Age"], errors="coerce").dropna()
            if len(dados_age_values) >= 20:
                age_values = pd.concat([age_values, dados_age_values], ignore_index=True)
        except (ValueError, OSError):
            pass

    return {
        "age_values": age_values.clip(18, 75),
        "gender_labels": gender_labels,
        "gender_probs": gender_probs,
        "package_labels": package_labels,
        "package_probs": package_probs,
        "workout_labels": workout_labels,
        "workout_probs": workout_probs,
        "duration_values": duration_values.clip(20, 150),
        "present_rate": float(np.clip(present_rate, 0.35, 0.85)),
        "reason_labels": reason_labels,
        "reason_probs": reason_probs,
    }


def sample_duration(values: pd.Series, rng: np.random.Generator) -> int:
    return int(np.clip(round(float(rng.choice(values.to_numpy()))), 20, 150))


def build_customers(
    n_customers: int, rng: np.random.Generator, refs: dict[str, object]
) -> tuple[pd.DataFrame, pd.Series]:
    scenarios = pd.Series(
        rng.choice(SCENARIOS, size=n_customers, p=SCENARIO_PROBABILITIES),
        name="scenario",
    )
    package_preferences = rng.choice(
        refs["package_labels"], size=n_customers, p=refs["package_probs"]
    )
    # Package-expiry risk is easier to express with a package that can end during the
    # prediction horizon, so this scenario uses monthly or quarterly membership.
    risk_mask = scenarios.eq("package_expiry_risk")
    package_preferences[risk_mask.to_numpy()] = rng.choice(
        ["Monthly", "Quarterly"], size=int(risk_mask.sum()), p=[0.70, 0.30]
    )

    customers = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
            "age": rng.choice(refs["age_values"].to_numpy(), size=n_customers).round().astype(int),
            "gender": rng.choice(
                refs["gender_labels"], size=n_customers, p=refs["gender_probs"]
            ),
            "package_type_preference": package_preferences,
            "preferred_visits_per_week": rng.choice(
                [1, 2, 3, 4], size=n_customers, p=[0.18, 0.34, 0.32, 0.16]
            ),
            "has_personal_training": rng.choice([0, 1], size=n_customers, p=[0.76, 0.24]),
            "prefers_group_lesson": rng.choice([0, 1], size=n_customers, p=[0.62, 0.38]),
            "avg_time_in_gym_minutes": np.clip(
                rng.normal(58, 17, size=n_customers).round().astype(int), 25, 130
            ),
            "coach_id": rng.choice(
                [f"COACH-{i:03d}" for i in range(1, 31)], size=n_customers
            ),
            "data_origin": DATA_ORIGIN,
        }
    )
    return customers, scenarios


def build_packages(
    customers: pd.DataFrame,
    scenarios: pd.Series,
    rng: np.random.Generator,
    period_start: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]]]:
    package_duration = {"Monthly": 30, "Quarterly": 90, "Annual": 365}
    package_rows: list[dict[str, object]] = []
    package_lookup: dict[str, list[dict[str, object]]] = {}
    package_number = 1

    for customer_row, scenario in zip(customers.itertuples(index=False), scenarios, strict=True):
        customer_id = customer_row.customer_id
        package_type = customer_row.package_type_preference
        if scenario == "package_expiry_risk":
            expiry_stop = cutoff_date + pd.Timedelta(days=int(rng.integers(5, 22)))
        else:
            expiry_stop = period_end

        cursor = period_start
        renewal_of = NONE_REF
        customer_packages: list[dict[str, object]] = []
        while cursor <= expiry_stop:
            nominal_end = cursor + pd.Timedelta(days=package_duration[package_type] - 1)
            end_date = min(nominal_end, expiry_stop)
            package_id = f"PKG-{package_number:05d}"
            package_number += 1
            total_sessions = int(
                max(
                    1,
                    round(
                        customer_row.preferred_visits_per_week
                        * ((end_date - cursor).days + 1)
                        / 7
                        * 1.15
                    ),
                )
            )
            status = "active" if end_date >= period_end else "renewed"
            if scenario == "package_expiry_risk" and end_date < period_end:
                status = "expired"
            package = {
                "package_id": package_id,
                "customer_id": customer_id,
                "package_type": package_type,
                "total_sessions": total_sessions,
                "start_date": cursor.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "status": status,
                "renewal_of": renewal_of,
                "data_origin": DATA_ORIGIN,
            }
            package_rows.append(package)
            customer_packages.append(package)
            renewal_of = package_id
            cursor = end_date + pd.Timedelta(days=1)
            if package_type == "Annual" or end_date >= period_end:
                break
        package_lookup[customer_id] = customer_packages

    return pd.DataFrame(package_rows), package_lookup


def find_package(
    package_list: list[dict[str, object]], date_value: pd.Timestamp
) -> dict[str, object] | None:
    for package in package_list:
        start = pd.Timestamp(package["start_date"])
        end = pd.Timestamp(package["end_date"])
        if start <= date_value <= end:
            return package
    return None


def attendance_probability(
    scenario: str,
    week_number: int,
    customer_effect: float,
    base_rate: float,
    rng: np.random.Generator,
) -> float:
    probability = base_rate + customer_effect
    if scenario == "stable_active":
        probability += 0.08
    elif scenario == "gradual_decline":
        probability += 0.10 - 0.035 * week_number
    elif scenario == "temporary_disruption":
        probability += 0.08 if week_number not in (5, 6) else -0.25
    elif scenario == "irregular_attendance":
        probability += rng.normal(-0.02, 0.13)
    elif scenario == "package_expiry_risk":
        probability += 0.04 if week_number < 9 else -0.07
    return float(np.clip(probability, 0.10, 0.95))


def build_sessions_and_events(
    customers: pd.DataFrame,
    scenarios: pd.Series,
    package_lookup: dict[str, list[dict[str, object]]],
    rng: np.random.Generator,
    refs: dict[str, object],
    period_start: pd.Timestamp,
    total_weeks: int,
    period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_rows: list[dict[str, object]] = []
    attendance_rows: list[dict[str, object]] = []
    reschedule_rows: list[dict[str, object]] = []
    session_number = 1
    attendance_number = 1
    reschedule_number = 1
    base_rate = float(np.clip(0.54 + 0.20 * refs["present_rate"], 0.60, 0.78))
    reason_labels = refs["reason_labels"]
    reason_probs = refs["reason_probs"]

    for customer_row, scenario in zip(customers.itertuples(index=False), scenarios, strict=True):
        customer_id = customer_row.customer_id
        customer_packages = package_lookup[customer_id]
        customer_effect = float(rng.normal(0, 0.06))
        visit_count = int(customer_row.preferred_visits_per_week)
        weekdays = np.sort(rng.choice(np.arange(7), size=visit_count, replace=False))

        for week_number in range(total_weeks):
            for weekday in weekdays:
                scheduled_date = period_start + pd.Timedelta(days=week_number * 7 + int(weekday))
                if scheduled_date > period_end:
                    continue
                package = find_package(customer_packages, scheduled_date)
                if package is None:
                    continue

                session_id = f"SES-{session_number:06d}"
                session_number += 1
                hour = int(rng.choice([6, 7, 9, 12, 17, 19]))
                minute = int(rng.choice([0, 15, 30, 45]))
                scheduled_at = scheduled_date + pd.Timedelta(hours=hour, minutes=minute)
                duration = sample_duration(refs["duration_values"], rng)
                session_type = str(
                    rng.choice(refs["workout_labels"], p=refs["workout_probs"])
                )

                reschedule_probability = {
                    "stable_active": 0.035,
                    "gradual_decline": 0.060 + 0.002 * week_number,
                    "temporary_disruption": 0.055 if week_number not in (5, 6) else 0.11,
                    "irregular_attendance": 0.085,
                    "package_expiry_risk": 0.105,
                }[scenario]
                candidate_shifts = [
                    shift for shift in [1, 2, 3, 7]
                    if scheduled_date + pd.Timedelta(days=shift)
                    <= pd.Timestamp(package["end_date"])
                ]
                will_reschedule = bool(
                    candidate_shifts and rng.random() < reschedule_probability
                )
                effective_at = scheduled_at
                session_status = "scheduled"

                if will_reschedule:
                    lead_time = float(rng.choice([4, 8, 12, 24, 48, 72]))
                    shift_days = int(rng.choice(candidate_shifts))
                    rescheduled_date = scheduled_date + pd.Timedelta(days=shift_days)
                    rescheduled_at = rescheduled_date + pd.Timedelta(hours=hour, minutes=minute)
                    requested_at = scheduled_at - pd.Timedelta(hours=lead_time)
                    initiated_by = str(rng.choice(["customer", "customer", "admin"]))
                    reason = str(rng.choice(reason_labels, p=reason_probs))
                    reschedule_rows.append(
                        {
                            "reschedule_id": f"RS-{reschedule_number:05d}",
                            "session_id": session_id,
                            "customer_id": customer_id,
                            "requested_at": requested_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "original_scheduled_at": scheduled_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "rescheduled_at": rescheduled_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "lead_time_hours": lead_time,
                            "reason": reason,
                            "initiated_by": initiated_by,
                            "data_origin": DATA_ORIGIN,
                        }
                    )
                    reschedule_number += 1
                    effective_at = rescheduled_at
                    session_status = "rescheduled"

                probability = attendance_probability(
                    scenario,
                    week_number,
                    customer_effect,
                    base_rate,
                    rng,
                )
                is_present = bool(rng.random() < probability)
                if is_present:
                    check_in = effective_at + pd.Timedelta(minutes=int(rng.integers(0, 11)))
                    actual_duration = int(np.clip(round(duration * rng.uniform(0.75, 1.10)), 15, 180))
                    check_out = check_in + pd.Timedelta(minutes=actual_duration)
                    check_in_value = check_in.strftime("%Y-%m-%d %H:%M:%S")
                    check_out_value = check_out.strftime("%Y-%m-%d %H:%M:%S")
                    session_status = "attended" if not will_reschedule else "rescheduled"
                else:
                    actual_duration = 0
                    check_in_value = SENTINEL
                    check_out_value = SENTINEL
                    session_status = "absent" if not will_reschedule else "rescheduled"

                session_rows.append(
                    {
                        "session_id": session_id,
                        "customer_id": customer_id,
                        "package_id": package["package_id"],
                        "scheduled_at": scheduled_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "session_type": session_type,
                        "planned_duration_minutes": duration,
                        "status": session_status,
                        "rescheduled_from": NONE_REF,
                        "data_origin": DATA_ORIGIN,
                    }
                )
                attendance_rows.append(
                    {
                        "attendance_id": f"ATT-{attendance_number:06d}",
                        "session_id": session_id,
                        "customer_id": customer_id,
                        "attendance_status": "present" if is_present else "absent",
                        "check_in_at": check_in_value,
                        "check_out_at": check_out_value,
                        "actual_duration_minutes": actual_duration,
                        "data_origin": DATA_ORIGIN,
                    }
                )
                attendance_number += 1

    return (
        pd.DataFrame(session_rows),
        pd.DataFrame(attendance_rows),
        pd.DataFrame(reschedule_rows),
    )


def build_admin_interactions(
    customers: pd.DataFrame,
    scenarios: pd.Series,
    rng: np.random.Generator,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    interaction_number = 1
    mean_by_scenario = {
        "stable_active": 1.3,
        "gradual_decline": 2.4,
        "temporary_disruption": 2.2,
        "irregular_attendance": 2.0,
        "package_expiry_risk": 2.7,
    }
    topics = ["membership", "schedule_change", "attendance_reminder", "payment", "feedback"]
    topic_probs = [0.22, 0.22, 0.28, 0.14, 0.14]
    channels = ["whatsapp", "phone", "email", "in_person"]
    outcomes = ["resolved", "responded", "no_response", "follow_up"]
    outcome_probs = [0.42, 0.34, 0.12, 0.12]

    for customer_row, scenario in zip(customers.itertuples(index=False), scenarios, strict=True):
        interaction_count = int(rng.poisson(mean_by_scenario[scenario]))
        for _ in range(interaction_count):
            span_days = max(1, (period_end - period_start).days + 1)
            occurred_at = period_start + pd.Timedelta(
                days=int(rng.integers(0, span_days)),
                hours=int(rng.integers(8, 21)),
                minutes=int(rng.choice([0, 15, 30, 45])),
            )
            direction = str(rng.choice(["inbound", "outbound"], p=[0.62, 0.38]))
            outcome = str(rng.choice(outcomes, p=outcome_probs))
            rows.append(
                {
                    "interaction_id": f"INT-{interaction_number:06d}",
                    "customer_id": customer_row.customer_id,
                    "occurred_at": occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "channel": str(rng.choice(channels, p=[0.55, 0.20, 0.15, 0.10])),
                    "direction": direction,
                    "topic": str(rng.choice(topics, p=topic_probs)),
                    "outcome": outcome,
                    "admin_id": str(rng.choice([f"ADMIN-{i:03d}" for i in range(1, 16)])),
                    "data_origin": DATA_ORIGIN,
                }
            )
            interaction_number += 1
    return pd.DataFrame(rows)


def assert_no_missing_or_duplicates(name: str, frame: pd.DataFrame, primary_key: str) -> None:
    if frame.empty:
        raise AssertionError(f"{name} is empty")
    if frame.isna().any().any():
        columns = frame.columns[frame.isna().any()].tolist()
        raise AssertionError(f"{name} contains missing values in {columns}")
    if frame.duplicated().any():
        raise AssertionError(f"{name} contains duplicate rows")
    if frame[primary_key].duplicated().any():
        raise AssertionError(f"{name}.{primary_key} is not unique")
    if frame[primary_key].astype(str).str.strip().eq("").any():
        raise AssertionError(f"{name}.{primary_key} contains empty keys")


def assert_foreign_keys(
    customers: pd.DataFrame,
    packages: pd.DataFrame,
    sessions: pd.DataFrame,
    attendance: pd.DataFrame,
    reschedules: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    customer_ids = set(customers["customer_id"])
    package_ids = set(packages["package_id"])
    session_ids = set(sessions["session_id"])
    assert set(packages["customer_id"]).issubset(customer_ids)
    assert set(sessions["customer_id"]).issubset(customer_ids)
    assert set(sessions["package_id"]).issubset(package_ids)
    assert set(attendance["session_id"]).issubset(session_ids)
    assert set(attendance["customer_id"]).issubset(customer_ids)
    assert set(reschedules["session_id"]).issubset(session_ids)
    assert set(reschedules["customer_id"]).issubset(customer_ids)
    assert set(interactions["customer_id"]).issubset(customer_ids)

    package_customer = packages.set_index("package_id")["customer_id"]
    session_customer = sessions.set_index("session_id")["customer_id"]
    assert (sessions["customer_id"] == sessions["package_id"].map(package_customer)).all()
    assert (attendance["customer_id"] == attendance["session_id"].map(session_customer)).all()
    assert (reschedules["customer_id"] == reschedules["session_id"].map(session_customer)).all()
    assert len(attendance) == len(sessions)
    assert attendance["session_id"].is_unique
    assert reschedules["session_id"].is_unique


def assert_temporal_consistency(
    packages: pd.DataFrame,
    sessions: pd.DataFrame,
    reschedules: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    package_start = pd.to_datetime(packages["start_date"])
    package_end = pd.to_datetime(packages["end_date"])
    assert (package_start <= package_end).all()
    session_time = pd.to_datetime(sessions["scheduled_at"])
    package_index = packages.set_index("package_id")
    starts = pd.to_datetime(sessions["package_id"].map(package_index["start_date"]))
    ends = pd.to_datetime(sessions["package_id"].map(package_index["end_date"]))
    assert ((session_time.dt.normalize() >= starts) & (session_time.dt.normalize() <= ends)).all()

    if not reschedules.empty:
        requested = pd.to_datetime(reschedules["requested_at"])
        original = pd.to_datetime(reschedules["original_scheduled_at"])
        moved = pd.to_datetime(reschedules["rescheduled_at"])
        assert (requested <= original).all()
        assert (moved > original).all()
        session_times = sessions.set_index("session_id")["scheduled_at"].map(pd.to_datetime)
        assert (original == reschedules["session_id"].map(session_times)).all()
        assert (reschedules["lead_time_hours"] == (original - requested).dt.total_seconds() / 3600).all()

    interaction_time = pd.to_datetime(interactions["occurred_at"])
    assert interaction_time.notna().all()


def validate_and_write(frames: dict[str, tuple[pd.DataFrame, str]], output_dir: Path) -> None:
    for name, (frame, primary_key) in frames.items():
        assert_no_missing_or_duplicates(name, frame, primary_key)

    customers = frames["customers.csv"][0]
    packages = frames["packages.csv"][0]
    sessions = frames["sessions.csv"][0]
    attendance = frames["attendance.csv"][0]
    reschedules = frames["reschedules.csv"][0]
    interactions = frames["admin_interactions.csv"][0]
    assert_foreign_keys(customers, packages, sessions, attendance, reschedules, interactions)
    assert_temporal_consistency(packages, sessions, reschedules, interactions)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (frame, _) in frames.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    # Re-read the exact files that will be consumed by the notebook. keep_default_na=False
    # ensures the explicit sentinel values remain non-missing.
    reread = {
        filename: pd.read_csv(output_dir / filename, keep_default_na=False)
        for filename in frames
    }
    for filename, frame in reread.items():
        assert_no_missing_or_duplicates(filename, frame, frames[filename][1])
    assert_foreign_keys(
        reread["customers.csv"],
        reread["packages.csv"],
        reread["sessions.csv"],
        reread["attendance.csv"],
        reread["reschedules.csv"],
        reread["admin_interactions.csv"],
    )
    assert_temporal_consistency(
        reread["packages.csv"],
        reread["sessions.csv"],
        reread["reschedules.csv"],
        reread["admin_interactions.csv"],
    )


def main() -> None:
    args = parse_args()
    if args.n_customers <= 0 or args.observation_weeks <= 0 or args.horizon_weeks <= 0:
        raise ValueError("n-customers, observation-weeks, and horizon-weeks must be positive")

    rng = np.random.default_rng(args.seed)
    refs = load_references(args.raw_dir)
    period_start = pd.Timestamp(args.start_date)
    cutoff_date = period_start + pd.Timedelta(weeks=args.observation_weeks)
    total_weeks = args.observation_weeks + args.horizon_weeks
    period_end = period_start + pd.Timedelta(weeks=total_weeks, days=-1)

    customers, scenarios = build_customers(args.n_customers, rng, refs)
    packages, package_lookup = build_packages(
        customers, scenarios, rng, period_start, cutoff_date, period_end
    )
    sessions, attendance, reschedules = build_sessions_and_events(
        customers,
        scenarios,
        package_lookup,
        rng,
        refs,
        period_start,
        total_weeks,
        period_end,
    )
    interactions = build_admin_interactions(
        customers, scenarios, rng, period_start, period_end
    )

    frames = {
        "customers.csv": (customers, "customer_id"),
        "packages.csv": (packages, "package_id"),
        "sessions.csv": (sessions, "session_id"),
        "attendance.csv": (attendance, "attendance_id"),
        "reschedules.csv": (reschedules, "reschedule_id"),
        "admin_interactions.csv": (interactions, "interaction_id"),
    }
    validate_and_write(frames, args.output_dir)

    print(f"Created six validated datasets in {args.output_dir}")
    print(f"Reference window: {period_start.date()} to {period_end.date()}")
    print(f"Prediction cutoff: {cutoff_date.date()}")
    for filename, (frame, _) in frames.items():
        print(f"- {filename}: {len(frame):,} rows")
    print("- validation: passed (no missing values, duplicate rows, broken foreign keys, or invalid event timestamps)")


if __name__ == "__main__":
    main()
