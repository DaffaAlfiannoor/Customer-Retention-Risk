"""Aturan skor prioritas yang dipakai bersama dashboard dan export frontend."""

from __future__ import annotations

from typing import Callable

import pandas as pd

# Setiap sinyal: (key, label, kolom, predikat). Urutan = urutan tampil di UI.
SIGNAL_RULES: list[tuple[str, str, str, Callable[[pd.Series], pd.Series]]] = [
    ("low_attendance", "Attendance below 60%", "attendance_rate", lambda s: s < 0.60),
    ("low_recent_attendance", "Recent attendance below 60%", "recent_attendance_rate", lambda s: s < 0.60),
    ("negative_trend", "Attendance trend is negative", "attendance_trend", lambda s: s < 0),
    ("absence_streak", "3+ consecutive absences", "max_consecutive_absences", lambda s: s >= 3),
    ("no_recent_visit", "No visit for 7+ days", "days_since_last_visit", lambda s: s >= 7),
    ("low_package_usage", "Package usage below 45%", "package_usage_ratio", lambda s: s < 0.45),
    ("high_reschedule", "Reschedule rate above 8%", "reschedule_rate", lambda s: s > 0.08),
    ("many_admin_contacts", "3+ admin contacts", "admin_contact_count", lambda s: s >= 3),
]


def signal_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Satu kolom boolean per sinyal, memakai predikat yang sama dengan skor."""
    return pd.DataFrame(
        {key: predicate(frame[column]).astype(bool) for key, _, column, predicate in SIGNAL_RULES},
        index=frame.index,
    )


def add_priority_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the transparent rule score used to rank customers for follow-up."""
    scored = frame.copy()
    scored["priority_score"] = signal_flags(scored).sum(axis=1).astype(int)
    scored["priority_band"] = pd.cut(
        scored["priority_score"],
        bins=[-0.1, 2, 5, 8],
        labels=["Rendah", "Sedang", "Tinggi"],
    )
    scored["usage_segment"] = pd.cut(
        scored["package_usage_ratio"],
        bins=[-0.1, 0.40, 0.70, 1.1],
        labels=["Rendah", "Sedang", "Tinggi"],
    )
    return scored
