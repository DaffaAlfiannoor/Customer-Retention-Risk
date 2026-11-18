"""Dashboard Customer Retention Risk Gymnasyum."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))
from scoring import add_priority_score  # noqa: E402  (satu sumber aturan skor)
DEFAULT_DATA_PATH = APP_DIR.parent / "data_ready" / "modeling_df.csv"

ID_COLUMN = "customer_id"
TARGET_COLUMN = "needs_follow_up"
FEATURE_COLUMNS = [
    "attendance_rate",
    "recent_attendance_rate",
    "attended_sessions",
    "missed_sessions",
    "visit_frequency",
    "attendance_trend",
    "max_consecutive_absences",
    "days_since_last_visit",
    "package_usage_ratio",
    "days_to_package_expiry",
    "renewal_count",
    "reschedule_count",
    "reschedule_rate",
    "admin_contact_count",
    "inbound_contact_count",
    "outbound_contact_count",
    "no_response_count",
    "follow_up_count",
    "resolved_count",
    "package_type",
]
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]

# Same semantics as the notebook, so dashboard and analysis never disagree.
COLOR_ALERT = "#e34948"
COLOR_SIGNAL = "#2a78d6"
COLOR_AMBER = "#b07000"
COLOR_INK = "#10161c"
COLOR_SOFT = "#55606b"
COLOR_RULE = "#dfe3de"
COLOR_FLOOR = "#151a20"

FOLLOW_UP_LABELS = {1: "Perlu follow-up", 0: "Tidak perlu follow-up"}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --floor: #151a20;
  --surface: #f5f6f4;
  --panel: #ffffff;
  --ink: #10161c;
  --soft: #55606b;
  --rule: #dfe3de;
  --signal: #2a78d6;
  --alert: #e34948;
  --amber: #b07000;
}

.stApp { background: var(--surface); }

html, body, [class*="css"], .stMarkdown, .stText, p, li, label {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--ink);
}

h1, h2, h3, h4, .masthead-title {
  font-family: 'Archivo', system-ui, sans-serif;
  letter-spacing: -0.02em;
  color: var(--ink);
}

.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }

/* Masthead */
.masthead {
  background: var(--floor);
  border-radius: 4px;
  padding: 1.5rem 1.75rem 1.4rem;
  position: relative;
  overflow: hidden;
}
.masthead::after {
  content: "";
  position: absolute; inset: 0;
  background: repeating-linear-gradient(
      115deg, rgba(255,255,255,.045) 0 1px, transparent 1px 26px);
  pointer-events: none;
}
.masthead-title {
  color: #ffffff; font-size: 1.35rem; font-weight: 700;
  margin: 0 0 .15rem; letter-spacing: -.01em;
}
.masthead-sub { color: #98a2ad; font-size: .78rem; margin: 0; }

.queue-line { display: flex; align-items: flex-end; gap: .9rem; margin-top: 1.35rem; }
.queue-number {
  font-family: 'Archivo', sans-serif; font-weight: 800;
  font-size: 3.6rem; line-height: .85; color: #ffffff;
  font-variant-numeric: tabular-nums;
}
.queue-label { color: #c9d1d9; font-size: .82rem; line-height: 1.35; padding-bottom: .35rem; }
.queue-label b { color: #ffffff; font-weight: 600; }

.queue-track {
  margin-top: 1.1rem; height: 14px; border-radius: 2px;
  background: rgba(255,255,255,.10); position: relative; overflow: hidden;
}
.queue-fill {
  height: 100%;
  background-image: repeating-linear-gradient(
      90deg, var(--alert) 0 9px, transparent 9px 11px);
}
.queue-caption {
  display: flex; justify-content: space-between;
  color: #7d8894; font-size: .72rem; margin-top: .5rem;
  font-variant-numeric: tabular-nums;
}

/* Tiles */
.tile-row { display: flex; gap: .75rem; margin: 1.1rem 0 .3rem; flex-wrap: wrap; }
.tile {
  flex: 1 1 150px; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 4px;
  padding: .8rem .9rem .75rem;
}
.tile-label { color: var(--soft); font-size: .72rem; margin: 0 0 .3rem; }
.tile-value {
  font-family: 'Archivo', sans-serif; font-weight: 700;
  font-size: 1.55rem; line-height: 1; margin: 0;
  font-variant-numeric: tabular-nums;
}
.tile-note { color: var(--soft); font-size: .7rem; margin: .3rem 0 0; }

/* Section headings */
.section-head {
  display: flex; align-items: baseline; gap: .6rem;
  border-bottom: 1px solid var(--rule);
  padding-bottom: .45rem; margin: 2rem 0 .9rem;
}
.section-head h2 { font-size: 1.02rem; font-weight: 600; margin: 0; }
.section-head span { color: var(--soft); font-size: .76rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-family: 'Archivo', sans-serif; font-size: .95rem; font-weight: 600;
}

/* Buttons */
.stDownloadButton button {
  background: var(--floor); color: #fff; border: none;
  border-radius: 3px; font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 500; font-size: .82rem; padding: .45rem 1rem;
}
.stDownloadButton button:hover { background: #262d36; color: #fff; }

.note { color: var(--soft); font-size: .74rem; line-height: 1.5; }
</style>
"""


@st.cache_data(show_spinner=False)
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    """Load the handoff dataset and fail loudly when columns are missing."""
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    return add_priority_score(frame)


def chart_frame(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure that sits flat on the dashboard panel."""
    fig, axis = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_alpha(0)
    axis.set_facecolor("none")
    return fig, axis


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=COLOR_RULE, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(COLOR_RULE)
    axis.tick_params(axis="both", length=0, labelsize=8, colors=COLOR_SOFT)
    axis.xaxis.label.set_color(COLOR_SOFT)
    axis.yaxis.label.set_color(COLOR_SOFT)


def share_bar(axis: plt.Axes, labels, values, title: str, xlabel: str) -> None:
    """Horizontal bar with direct labels and a quiet grid."""
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=COLOR_ALERT, height=0.58, zorder=3)
    axis.bar_label(bars, fmt="%.0f%%", padding=4, color=COLOR_INK, fontsize=8)
    axis.set_yticks(positions, labels=labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(0, max(float(values.max()) * 1.3, 10))
    axis.set_title(title, loc="left", fontweight="bold", fontsize=9.5,
                   color=COLOR_INK, pad=10)
    axis.set_xlabel(xlabel, fontsize=8)
    style_axis(axis, grid_axis="x")


def follow_up_share(frame: pd.DataFrame, group_column: str) -> pd.Series:
    return frame.groupby(group_column, observed=False)[TARGET_COLUMN].mean().mul(100)


st.set_page_config(
    page_title="Retention Desk · Gymnasyum",
    page_icon="🏋",
    layout="wide",
)
st.markdown(STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Sumber data")
    uploaded_file = st.file_uploader(
        "Unggah modeling_df.csv",
        type="csv",
        help="Format kolom harus sama dengan data_ready/modeling_df.csv.",
    )

    if uploaded_file is not None:
        raw_frame = pd.read_csv(uploaded_file)
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in raw_frame.columns
        ]
        if missing_columns:
            st.error(f"Kolom wajib tidak ditemukan: {missing_columns}")
            st.stop()
        data = add_priority_score(raw_frame)
        st.caption(f"Memakai unggahan · {len(data):,} customer")
    else:
        if not DEFAULT_DATA_PATH.exists():
            st.error(f"File default tidak ditemukan: {DEFAULT_DATA_PATH}")
            st.stop()
        data = load_data(str(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.stat().st_mtime)
        st.caption(f"Memakai data_ready · {len(data):,} customer")

    st.markdown("## Filter")
    package_options = sorted(data["package_type"].dropna().unique())
    selected_packages = st.multiselect(
        "Tipe package", options=package_options, default=package_options
    )
    attendance_range = st.slider(
        "Attendance rate historis", 0.0, 1.0, (0.0, 1.0), 0.05
    )
    recent_range = st.slider(
        "Recent attendance (30 hari)", 0.0, 1.0, (0.0, 1.0), 0.05
    )
    max_days_since_visit = st.slider(
        "Maksimum hari sejak kunjungan terakhir",
        int(data["days_since_last_visit"].min()),
        int(data["days_since_last_visit"].max()),
        int(data["days_since_last_visit"].max()),
    )
    min_priority_score = st.slider(
        "Minimum sinyal aturan", 0, int(data["priority_score"].max()), 0
    )
    follow_up_only = st.checkbox("Hanya customer perlu follow-up", value=False)

    st.markdown(
        '<p class="note">Skor prioritas dihitung dari delapan sinyal historis: '
        "attendance, tren, absensi berurutan, jarak kunjungan, package usage, "
        "reschedule, dan interaksi admin.</p>",
        unsafe_allow_html=True,
    )

filtered = data.loc[
    data["package_type"].isin(selected_packages)
    & data["attendance_rate"].between(*attendance_range)
    & data["recent_attendance_rate"].between(*recent_range)
    & data["days_since_last_visit"].le(max_days_since_visit)
    & data["priority_score"].ge(min_priority_score)
]
if follow_up_only:
    filtered = filtered.loc[filtered[TARGET_COLUMN].eq(1)]

if filtered.empty:
    st.markdown("## Tidak ada customer pada filter ini")
    st.markdown(
        '<p class="note">Longgarkan rentang attendance atau turunkan minimum '
        "sinyal aturan di panel kiri untuk melihat customer lagi.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

total_customers = len(filtered)
follow_up_customers = int(filtered[TARGET_COLUMN].sum())
follow_up_share_pct = follow_up_customers / total_customers * 100
mean_signals = float(filtered["priority_score"].mean())

st.markdown(
    f"""
<div class="masthead">
  <p class="masthead-title">Retention Desk</p>
  <p class="masthead-sub">Gymnasyum · data sintetis · cutoff 31 Maret 2025 ·
  horizon 1–27 April 2025</p>
  <div class="queue-line">
    <div class="queue-number">{follow_up_customers:,}</div>
    <div class="queue-label">customer perlu follow-up<br>
      <b>dari {total_customers:,} customer</b> pada filter aktif</div>
  </div>
  <div class="queue-track">
    <div class="queue-fill" style="width:{follow_up_share_pct:.1f}%"></div>
  </div>
  <div class="queue-caption">
    <span>{follow_up_share_pct:.1f}% antrean follow-up</span>
    <span>{100 - follow_up_share_pct:.1f}% terpantau aman</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="tile-row">
  <div class="tile">
    <p class="tile-label">Attendance rate</p>
    <p class="tile-value">{filtered['attendance_rate'].mean():.0%}</p>
    <p class="tile-note">seluruh histori sampai cutoff</p>
  </div>
  <div class="tile">
    <p class="tile-label">Recent attendance</p>
    <p class="tile-value">{filtered['recent_attendance_rate'].mean():.0%}</p>
    <p class="tile-note">30 hari terakhir sebelum cutoff</p>
  </div>
  <div class="tile">
    <p class="tile-label">Package usage</p>
    <p class="tile-value">{filtered['package_usage_ratio'].mean():.0%}</p>
    <p class="tile-note">pemakaian sesi package aktif</p>
  </div>
  <div class="tile">
    <p class="tile-label">Rata-rata sinyal</p>
    <p class="tile-value">{mean_signals:.1f}<span style="font-size:.9rem;color:#55606b">/8</span></p>
    <p class="tile-note">jumlah aturan yang terpenuhi</p>
  </div>
  <div class="tile">
    <p class="tile-label">Jarak kunjungan</p>
    <p class="tile-value">{filtered['days_since_last_visit'].median():.0f}<span style="font-size:.9rem;color:#55606b"> hari</span></p>
    <p class="tile-note">median sejak kunjungan terakhir</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-head"><h2>Di mana antrean terkonsentrasi</h2>'
    "<span>proporsi customer perlu follow-up per segmen</span></div>",
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    fig, axis = chart_frame((6.4, 3.0))
    package_share = follow_up_share(filtered, "package_type").sort_values()
    share_bar(
        axis,
        package_share.index.astype(str),
        package_share.values,
        "Tipe package",
        "customer perlu follow-up (%)",
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    fig, axis = chart_frame((6.4, 3.0))
    usage_share = (
        follow_up_share(filtered, "usage_segment")
        .reindex(["Rendah", "Sedang", "Tinggi"])
        .dropna()
    )
    share_bar(
        axis,
        usage_share.index.astype(str),
        usage_share.values,
        "Package usage",
        "customer perlu follow-up (%)",
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns([1.15, 1])

with left:
    fig, axis = chart_frame((6.0, 3.0))
    profile = (
        filtered.groupby(TARGET_COLUMN)[
            ["attendance_rate", "recent_attendance_rate"]
        ]
        .mean()
        .mul(100)
        .reindex([1, 0])
    )
    positions = np.arange(len(profile))
    width = 0.34
    first = axis.bar(
        positions - width / 2,
        profile["attendance_rate"],
        width=width,
        color=COLOR_SIGNAL,
        label="Historis",
        zorder=3,
    )
    second = axis.bar(
        positions + width / 2,
        profile["recent_attendance_rate"],
        width=width,
        color=COLOR_AMBER,
        label="30 hari terakhir",
        zorder=3,
    )
    axis.bar_label(first, fmt="%.0f%%", padding=3, fontsize=8, color=COLOR_INK)
    axis.bar_label(second, fmt="%.0f%%", padding=3, fontsize=8, color=COLOR_INK)
    axis.set_xticks(positions, ["Perlu follow-up", "Aman"], fontsize=8.5)
    axis.set_ylim(0, 100)
    axis.set_title(
        "Attendance menurut kebutuhan follow-up",
        loc="left", fontweight="bold", fontsize=9.5, color=COLOR_INK, pad=10,
    )
    axis.set_ylabel("%", fontsize=8)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    style_axis(axis)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    fig, axis = chart_frame((5.2, 3.0))
    band_counts = (
        filtered["priority_band"]
        .value_counts()
        .reindex(["Rendah", "Sedang", "Tinggi"])
        .fillna(0)
    )
    band_colors = ["#c9d1d9", COLOR_AMBER, COLOR_ALERT]
    bars = axis.bar(
        band_counts.index.astype(str),
        band_counts.values,
        color=band_colors,
        width=0.55,
        zorder=3,
    )
    axis.bar_label(
        bars,
        labels=[f"{int(value):,}" for value in band_counts.values],
        padding=4,
        fontsize=8,
        color=COLOR_INK,
    )
    axis.set_ylim(0, max(float(band_counts.max()) * 1.2, 1))
    axis.set_title(
        "Sebaran skor prioritas",
        loc="left", fontweight="bold", fontsize=9.5, color=COLOR_INK, pad=10,
    )
    axis.set_ylabel("Jumlah customer", fontsize=8)
    style_axis(axis)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown(
    '<div class="section-head"><h2>Daftar kerja admin</h2>'
    "<span>urut berdasarkan sinyal aturan, bukan output model</span></div>",
    unsafe_allow_html=True,
)

priority_columns = [
    ID_COLUMN,
    "priority_score",
    "priority_band",
    "package_type",
    TARGET_COLUMN,
    "attendance_rate",
    "recent_attendance_rate",
    "attendance_trend",
    "max_consecutive_absences",
    "days_since_last_visit",
    "package_usage_ratio",
    "reschedule_rate",
    "admin_contact_count",
]

control, spacer = st.columns([1, 2.2])
with control:
    top_n = st.slider("Baris ditampilkan", 10, 200, 50, step=10)

priority_table = (
    filtered.loc[filtered[TARGET_COLUMN].eq(1), priority_columns]
    .sort_values(
        ["priority_score", "recent_attendance_rate", "days_since_last_visit"],
        ascending=[False, True, False],
    )
    .head(top_n)
    .reset_index(drop=True)
)

if priority_table.empty:
    st.markdown(
        '<p class="note">Tidak ada customer perlu follow-up pada filter ini. '
        "Matikan opsi “Hanya customer perlu follow-up” atau longgarkan filter.</p>",
        unsafe_allow_html=True,
    )
else:
    st.dataframe(
        priority_table.style.format(
            {
                "attendance_rate": "{:.2f}",
                "recent_attendance_rate": "{:.2f}",
                "attendance_trend": "{:+.2f}",
                "package_usage_ratio": "{:.2f}",
                "reschedule_rate": "{:.2f}",
            }
        ),
        width="stretch",
        height=430,
        hide_index=True,
    )
    st.download_button(
        "Unduh daftar prioritas",
        data=priority_table.to_csv(index=False).encode("utf-8"),
        file_name="daftar_prioritas_follow_up.csv",
        mime="text/csv",
    )

st.markdown(
    '<p class="note" style="margin-top:2rem">Data sintetis untuk prototype. '
    "Skor dan daftar prioritas adalah alat bantu; keputusan follow-up tetap "
    "memerlukan peninjauan admin.</p>",
    unsafe_allow_html=True,
)
