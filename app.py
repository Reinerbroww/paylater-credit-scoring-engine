import math

import altair as alt
import pandas as pd
import streamlit as st

from src.config import (
    APP_DEFAULTS,
    DEFAULT_CODES,
    FEATURE_ORDER,
    GRADE_CODES,
    HOME_CODES,
    INTENT_CODES,
    MODEL_METRICS,
    SAMPLE_PROFILES,
)
from src.pipeline import load_artifacts, score_applicant

DISPLAY_NAMES = {
    "en": {
        "person_age": "Customer Age",
        "person_income": "Annual Income",
        "person_emp_length": "Employment Length",
        "loan_amnt": "Requested Loan",
        "loan_int_rate": "Interest Rate",
        "loan_percent_income": "Loan % of Income",
        "cb_person_cred_hist_length": "Credit History Length",
        "person_home_ownership_OTHER": "Home: Other",
        "person_home_ownership_OWN": "Home: Own",
        "person_home_ownership_RENT": "Home: Rent",
        "loan_intent_EDUCATION": "Intent: Education",
        "loan_intent_HOMEIMPROVEMENT": "Intent: Home Improvement",
        "loan_intent_MEDICAL": "Intent: Medical",
        "loan_intent_PERSONAL": "Intent: Personal",
        "loan_intent_VENTURE": "Intent: Venture",
        "loan_grade_B": "Loan Grade B",
        "loan_grade_C": "Loan Grade C",
        "loan_grade_D": "Loan Grade D",
        "loan_grade_E": "Loan Grade E",
        "loan_grade_F": "Loan Grade F",
        "loan_grade_G": "Loan Grade G",
        "cb_person_default_on_file_Y": "Prior Default On File",
    },
    "id": {
        "person_age": "Usia Pelanggan",
        "person_income": "Pendapatan Tahunan",
        "person_emp_length": "Lama Bekerja",
        "loan_amnt": "Pinjaman Diminta",
        "loan_int_rate": "Suku Bunga",
        "loan_percent_income": "Pinjaman % Pendapatan",
        "cb_person_cred_hist_length": "Lama Riwayat Kredit",
        "person_home_ownership_OTHER": "Rumah: Lainnya",
        "person_home_ownership_OWN": "Rumah: Milik Sendiri",
        "person_home_ownership_RENT": "Rumah: Sewa",
        "loan_intent_EDUCATION": "Tujuan: Pendidikan",
        "loan_intent_HOMEIMPROVEMENT": "Tujuan: Renovasi Rumah",
        "loan_intent_MEDICAL": "Tujuan: Medis",
        "loan_intent_PERSONAL": "Tujuan: Pribadi",
        "loan_intent_VENTURE": "Tujuan: Usaha",
        "loan_grade_B": "Peringkat Pinjaman B",
        "loan_grade_C": "Peringkat Pinjaman C",
        "loan_grade_D": "Peringkat Pinjaman D",
        "loan_grade_E": "Peringkat Pinjaman E",
        "loan_grade_F": "Peringkat Pinjaman F",
        "loan_grade_G": "Peringkat Pinjaman G",
        "cb_person_default_on_file_Y": "Riwayat Gagal Bayar",
    },
}

HOME_LABELS = {
    "RENT": {"en": "Rent", "id": "Sewa"},
    "OWN": {"en": "Own", "id": "Milik Sendiri"},
    "MORTGAGE": {"en": "Mortgage", "id": "KPR / Kredit Rumah"},
    "OTHER": {"en": "Other", "id": "Lainnya"},
}

INTENT_LABELS = {
    "DEBTCONSOLIDATION": {"en": "Debt Consolidation", "id": "Konsolidasi Utang"},
    "EDUCATION": {"en": "Education", "id": "Pendidikan"},
    "HOMEIMPROVEMENT": {"en": "Home Improvement", "id": "Renovasi Rumah"},
    "MEDICAL": {"en": "Medical", "id": "Medis"},
    "PERSONAL": {"en": "Personal", "id": "Pribadi"},
    "VENTURE": {"en": "Venture", "id": "Usaha"},
}

DEFAULT_LABELS = {"N": {"en": "No", "id": "Tidak"}, "Y": {"en": "Yes", "id": "Ya"}}

LANG_LABELS = {"en": "English", "id": "Bahasa Indonesia"}

TEXTS = {
    "brand_title": {"en": "E-Commerce PayLater Credit Scoring", "id": "Penilaian Kredit PayLater E-Commerce"},
    "brand_sub": {
        "en": "Real-time risk assessment and credit decision engine",
        "id": "Penilaian risiko real-time dan mesin keputusan kredit",
    },
    "status_loaded": {
        "en": "Production LightGBM champion model loaded",
        "id": "Model juara LightGBM produksi berhasil dimuat",
    },
    "load_error": {"en": "Failed to load model artifacts: {error}", "id": "Gagal memuat artefak model: {error}"},

    "language": {"en": "Language", "id": "Bahasa"},
    "section_profiles": {"en": "Sample Profiles", "id": "Contoh Profil"},
    "btn_low": {"en": "🟢 Low risk applicant", "id": "🟢 Pelamar risiko rendah"},
    "btn_border": {"en": "🟡 Borderline applicant", "id": "🟡 Pelamar batas"},
    "btn_high": {"en": "🔴 High risk applicant", "id": "🔴 Pelamar risiko tinggi"},
    "section_thresholds": {"en": "Decision Thresholds", "id": "Ambang Keputusan"},
    "slider_low": {"en": "Approve below", "id": "Setujui di bawah"},
    "slider_high": {"en": "Reject above", "id": "Tolak di atas"},
    "section_champion": {"en": "Champion Model", "id": "Model Juara"},
    "champion_caption": {
        "en": "Leaf-wise gradient boosting · 22 engineered features",
        "id": "Leaf-wise gradient boosting · 22 fitur hasil rekayasa",
    },
    "precision": {"en": "Precision", "id": "Presisi"},
    "recall": {"en": "Recall", "id": "Recall"},
    "accuracy": {"en": "Accuracy", "id": "Akurasi"},
    "champion_note": {
        "en": "Benchmarked on the held-out validation set.",
        "id": "Diuji pada set validasi yang disisihkan.",
    },

    "threshold_warning": {
        "en": "Approve threshold must sit below the reject threshold. Adjusting the reject boundary.",
        "id": "Ambang setujui harus di bawah ambang tolak. Menyesuaikan batas penolakan.",
    },

    "section_profile": {"en": "Customer Financial Profile", "id": "Profil Keuangan Pelanggan"},
    "field_income": {"en": "Annual Income ($)", "id": "Pendapatan Tahunan ($)"},
    "field_loan": {"en": "Requested PayLater Loan ($)", "id": "Pinjaman PayLater Diminta ($)"},
    "field_age": {"en": "Customer Age", "id": "Usia Pelanggan"},
    "field_rate": {"en": "Assigned Interest Rate (%)", "id": "Suku Bunga Ditetapkan (%)"},
    "field_emp": {"en": "Employment Length (Years)", "id": "Lama Bekerja (Tahun)"},
    "field_home": {"en": "Home Ownership Status", "id": "Status Kepemilikan Rumah"},
    "section_intent": {"en": "Loan Intent Classification", "id": "Klasifikasi Tujuan Pinjaman"},
    "field_intent": {"en": "Loan Intent", "id": "Tujuan Pinjaman"},
    "expander_advanced": {"en": "Advanced risk attributes", "id": "Atribut risiko lanjutan"},
    "field_grade": {"en": "Loan Grade", "id": "Peringkat Pinjaman"},
    "field_default": {"en": "Prior Default On File", "id": "Riwayat Gagal Bayar"},
    "field_hist": {"en": "Credit History Length (Years)", "id": "Lama Riwayat Kredit (Tahun)"},
    "submit": {"en": "Evaluate PayLater Risk", "id": "Evaluasi Risiko PayLater"},

    "section_eval": {"en": "Evaluation Summary", "id": "Ringkasan Evaluasi"},
    "info_prompt": {
        "en": "Fill in the customer profile above and press **Evaluate PayLater Risk** to generate a decision.",
        "id": "Isi profil pelanggan di atas lalu tekan **Evaluasi Risiko PayLater** untuk membuat keputusan.",
    },

    "section_context": {"en": "Dynamic Underwriting Context", "id": "Konteks Underwriting Dinamis"},
    "card_exposure": {"en": "Paycheck Exposure", "id": "Paparan Gaji"},
    "card_exposure_sub": {"en": "Requested loan / annual income", "id": "Pinjaman diminta / pendapatan tahunan"},
    "card_segment": {"en": "Model Segment", "id": "Segmen Model"},
    "card_segment_sub": {"en": "Leaf-wise boosting core", "id": "Inti leaf-wise boosting"},
    "card_vector": {"en": "Feature Vector", "id": "Vektor Fitur"},
    "card_vector_sub": {"en": "Numeric + one-hot encoded", "id": "Numerik + one-hot encoding"},
    "card_scaling": {"en": "Scaling", "id": "Penskalaan"},
    "card_scaling_sub": {"en": "Fitted preprocessing binary", "id": "Biner prapemrosesan terlatih"},

    "expander_why": {
        "en": "Why this decision — feature contribution breakdown",
        "id": "Mengapa keputusan ini — rincian kontribusi fitur",
    },
    "contrib_x": {"en": "Impact on risk score (log-odds)", "id": "Dampak pada skor risiko (log-odds)"},
    "contrib_caption": {
        "en": "Red pushes the applicant toward default risk; green reduces it. Relative influence only.",
        "id": "Merah mendorong pelamar ke risiko gagal bayar; hijau menurunkannya. Hanya pengaruh relatif.",
    },
    "expander_how": {"en": "How this engine works", "id": "Cara kerja mesin ini"},
    "how_body": {
        "en": (
            "1. **Input capture** — applicant financial attributes are collected from the profile form.\n"
            "2. **Feature engineering** — `loan_percent_income` is derived, then categorical fields are mapped "
            "to their exact one-hot encoded counterparts used during training.\n"
            "3. **Normalization** — the vector passes through the pre-fitted `credit_data_scaler.pkl`.\n"
            "4. **Inference** — the serialized LightGBM champion returns a default probability.\n"
            "5. **Decision gate** — the probability is routed to the approve / manual review / reject bands."
        ),
        "id": (
            "1. **Input data** — atribut keuangan pelamar diambil dari formulir profil.\n"
            "2. **Rekayasa fitur** — `loan_percent_income` dihitung, lalu kolom kategorikal dipetakan "
            "ke bentuk one-hot encoding yang sama seperti saat pelatihan.\n"
            "3. **Normalisasi** — vektor melewati `credit_data_scaler.pkl` yang sudah terlatih.\n"
            "4. **Inferensi** — model juara LightGBM menghasilkan probabilitas gagal bayar.\n"
            "5. **Ambang keputusan** — probabilitas diarahkan ke pita setujui / tinjau manual / tolak."
        ),
    },
    "footer_1": {
        "en": "Decision-support prototype. Outputs are model estimates and do not constitute a final credit decision.",
        "id": "Prototipe pendukung keputusan. Keluaran berupa estimasi model dan bukan keputusan kredit final.",
    },
    "footer_2": {"en": "Built with Streamlit · LightGBM · scikit-learn", "id": "Dibangun dengan Streamlit · LightGBM · scikit-learn"},

    "dec_approve_title": {"en": "✅ Application Approved", "id": "✅ Pengajuan Disetujui"},
    "dec_approve_rat": {
        "en": "Safe limit. Profile sits below the default threshold.",
        "id": "Batas aman. Profil berada di bawah ambang gagal bayar.",
    },
    "dec_review_title": {"en": "⚠️ Manual Review Required", "id": "⚠️ Perlu Tinjauan Manual"},
    "dec_review_rat": {
        "en": "Instant checkout halted. Routing vector traits to underwriting.",
        "id": "Checkout instan dihentikan. Mengarahkan atribut ke underwriting.",
    },
    "dec_reject_title": {"en": "⛔ Application Rejected", "id": "⛔ Pengajuan Ditolak"},
    "dec_reject_rat": {
        "en": "Checkout blocked to prevent credit loss.",
        "id": "Checkout diblokir untuk mencegah kerugian kredit.",
    },
    "dec_score": {"en": "Risk Probability Score:", "id": "Skor Probabilitas Risiko:"},
    "band_low": {"en": "Low", "id": "Rendah"},
    "band_medium": {"en": "Medium", "id": "Sedang"},
    "band_high": {"en": "High", "id": "Tinggi"},
    "dec_risk_band": {"en": "{band} risk band", "id": "pita risiko {band}"},
    "dec_ratio": {
        "en": "Automated check — ratio of obligation: <b>{value}%</b> of income.",
        "id": "Pemeriksaan otomatis — rasio kewajiban: <b>{value}%</b> dari pendapatan.",
    },
    "dec_gate": {
        "en": "Decision gate — approve &lt; {low}% · review {low}-{high}% · reject &gt; {high}%.",
        "id": "Ambang keputusan — setujui &lt; {low}% · tinjau {low}-{high}% · tolak &gt; {high}%.",
    },

    "gauge_caption": {"en": "DEFAULT PROBABILITY", "id": "PROBABILITAS GAGAL BAYAR"},
    "gauge_approve": {"en": "Approve < {p}%", "id": "Setujui < {p}%"},
    "gauge_review": {"en": "Review {low}-{high}%", "id": "Tinjau {low}-{high}%"},
    "gauge_reject": {"en": "Reject > {p}%", "id": "Tolak > {p}%"},
}

CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1180px;}

.topbar {
    background: #0f172a;
    border-radius: 14px;
    padding: 28px 32px;
    color: #ffffff;
    border-bottom: 5px solid #2563eb;
}
.topbar h1 {font-size: 1.95rem; margin: 0; font-weight: 800; letter-spacing: -0.02em;}
.topbar p {margin: 0.4rem 0 0; color: #94a3b8; font-size: 1rem; font-weight: 500;}
.status {
    display: inline-flex; align-items: center; gap: 0.5rem;
    margin-top: 1.05rem; font-size: 0.82rem; font-weight: 600;
    color: #4ade80; letter-spacing: 0.01em;
}
.status .dot {width: 0.55rem; height: 0.55rem; border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 0 3px rgba(34,197,94,.25);}

.section-title {
    display: flex; align-items: center; gap: 0.6rem;
    font-weight: 800; font-size: 1rem; text-transform: uppercase;
    letter-spacing: 0.06em; margin: 1.4rem 0 0.55rem; color: #0f172a;
}
.section-title .bar {width: 4px; height: 18px; border-radius: 2px; background: #2563eb;}

.decision-card {border: 2px solid #e2e8f0; border-radius: 14px; padding: 22px 24px; background: #ffffff;}
.dec-approve {border-left: 7px solid #16a34a;}
.dec-review {border-left: 7px solid #d97706;}
.dec-reject {border-left: 7px solid #dc2626;}
.decision-label {font-size: 1.3rem; font-weight: 800; letter-spacing: -0.01em;}
.decision-sub {color: #475569; font-size: 0.95rem; margin-top: 0.35rem; font-weight: 500;}
.dec-list {margin: 1rem 0 0; padding-left: 1.1rem; color: #334155; font-size: 0.92rem;}
.dec-list li {margin-bottom: 0.35rem;}

.metric-card {background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 12px;
    padding: 16px 18px; height: 100%;}
.metric-card .k {color: #64748b; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;}
.metric-card .v {color: #0f172a; font-size: 1.5rem; font-weight: 800; margin-top: 0.2rem;}
.metric-card .s {color: #94a3b8; font-size: 0.78rem; margin-top: 0.15rem;}

.gauge-wrap {display: flex; justify-content: center; padding: 0.4rem 0 0;}
.legend {display: flex; justify-content: center; gap: 1.4rem; margin-top: 0.4rem;
    font-size: 0.8rem; color: #475569; font-weight: 600;}
.legend span {display: inline-flex; align-items: center; gap: 0.4rem;}
.legend i {width: 0.7rem; height: 0.7rem; border-radius: 3px; display: inline-block;}

.badge {display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px;
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;}
.b-safe {background: rgba(22,163,74,.14); color: #15803d;}
.b-warn {background: rgba(217,119,6,.16); color: #b45309;}
.b-risk {background: rgba(220,38,38,.14); color: #b91c1c;}

div[data-testid="stFormSubmitButton"] button {
    background: #2563eb; color: #ffffff; border: none; font-weight: 700;
    font-size: 1rem; padding: 0.7rem 1rem; border-radius: 10px;
}
div[data-testid="stFormSubmitButton"] button:hover {background: #1d4ed8;}
</style>
"""


def t(key, **kwargs):
    lang = st.session_state.get("lang", "en")
    entry = TEXTS.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def current_lang():
    return st.session_state.get("lang", "en")


def label_for(mapping, code):
    return mapping[code][current_lang()]


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


def _gauge_point(cx, cy, r, theta_deg):
    t_rad = math.radians(theta_deg)
    return cx + r * math.cos(t_rad), cy - r * math.sin(t_rad)


def render_gauge(probability, low, high, stroke_color):
    cx, cy, r = 160.0, 152.0, 118.0
    zones = [(0.0, low, "#16a34a"), (low, high, "#d97706"), (high, 1.0, "#dc2626")]
    arcs = []
    for p0, p1, color in zones:
        if p1 <= p0:
            continue
        t0 = 180.0 * (1.0 - p0)
        t1 = 180.0 * (1.0 - p1)
        x0, y0 = _gauge_point(cx, cy, r, t0)
        x1, y1 = _gauge_point(cx, cy, r, t1)
        large = 1 if (t0 - t1) > 180 else 0
        arcs.append(
            f'<path d="M {x0:.2f} {y0:.2f} A {r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}" '
            f'stroke="{color}" stroke-width="24" fill="none" stroke-linecap="butt"/>'
        )

    theta = 180.0 * (1.0 - probability)
    nx, ny = _gauge_point(cx, cy, r - 30.0, theta)
    markers = []
    for boundary in (low, high):
        theta_b = 180.0 * (1.0 - boundary)
        mx0, my0 = _gauge_point(cx, cy, r - 12.0, theta_b)
        mx1, my1 = _gauge_point(cx, cy, r + 12.0, theta_b)
        markers.append(
            f'<line x1="{mx0:.2f}" y1="{my0:.2f}" x2="{mx1:.2f}" y2="{my1:.2f}" '
            f'stroke="#0f172a" stroke-width="2" stroke-dasharray="3 3" opacity="0.55"/>'
        )

    return f"""
    <div class="gauge-wrap">
    <svg viewBox="0 0 320 190" width="330" height="196" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
                <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.18"/>
            </filter>
        </defs>
        <g filter="url(#soft)">{''.join(arcs)}</g>
        {''.join(markers)}
        <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{stroke_color}"
              stroke-width="5" stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="11" fill="#0f172a"/>
        <circle cx="{cx}" cy="{cy}" r="4.5" fill="#ffffff"/>
        <text x="{cx}" y="{cy - 26}" text-anchor="middle" font-size="34"
              font-weight="800" fill="#0f172a">{probability * 100:.1f}%</text>
        <text x="{cx}" y="{cy - 6}" text-anchor="middle" font-size="11.5"
              font-weight="600" fill="#64748b">{t('gauge_caption')}</text>
    </svg>
    </div>
    <div class="legend">
        <span><i style="background:#16a34a"></i>{t('gauge_approve', p=f'{low * 100:.0f}')}</span>
        <span><i style="background:#d97706"></i>{t('gauge_review', low=f'{low * 100:.0f}', high=f'{high * 100:.0f}')}</span>
        <span><i style="background:#dc2626"></i>{t('gauge_reject', p=f'{high * 100:.0f}')}</span>
    </div>
    """


def render_contributions(model, contributions, top_n=8):
    names = DISPLAY_NAMES[current_lang()]
    frame = pd.DataFrame(
        {
            "Feature": [names.get(n, n) for n in FEATURE_ORDER],
            "Impact": contributions,
        }
    )
    frame["Magnitude"] = frame["Impact"].abs()
    frame = frame.sort_values("Magnitude", ascending=False).head(top_n).reset_index(drop=True)
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=5, height=20)
        .encode(
            x=alt.X("Impact:Q", title=t("contrib_x")),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.condition(alt.datum.Impact > 0, alt.value("#dc2626"), alt.value("#16a34a")),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Impact:Q", format=".3f"),
            ],
        )
        .properties(height=max(200, top_n * 30))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(t("contrib_caption"))


def apply_preset(profile):
    for key, value in profile.items():
        st.session_state[key] = value


def render_decision(probability, decision, ratio, low, high):
    config = {
        "approve": ("dec-approve", "dec_approve_title", "dec_approve_rat", "#15803d", "b-safe", "band_low"),
        "review": ("dec-review", "dec_review_title", "dec_review_rat", "#b45309", "b-warn", "band_medium"),
        "reject": ("dec-reject", "dec_reject_title", "dec_reject_rat", "#b91c1c", "b-risk", "band_high"),
    }
    css_class, title_key, rat_key, color, badge, band_key = config[decision]
    band = t(band_key)

    st.markdown(
        f"""
        <div class="decision-card {css_class}">
            <div class="decision-label" style="color:{color}">{t(title_key)}</div>
            <div class="decision-sub">{t('dec_score')} <b>{probability * 100:.2f}%</b>
                &nbsp;·&nbsp; <span class="badge {badge}">{t('dec_risk_band', band=band)}</span></div>
            <ul class="dec-list">
                <li>{t(rat_key)}</li>
                <li>{t('dec_ratio', value=f'{ratio * 100:.2f}')}</li>
                <li>{t('dec_gate', low=f'{low * 100:.0f}', high=f'{high * 100:.0f}')}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.radio("Language / Bahasa", ["en", "id"], format_func=lambda c: LANG_LABELS[c],
                 horizontal=True, key="lang")

        st.markdown(f'<div class="section-title" style="margin-top:1rem"><span class="bar"></span>{t("section_profiles")}</div>', unsafe_allow_html=True)
        st.button(t("btn_low"), width="stretch", on_click=apply_preset, args=(SAMPLE_PROFILES["Low risk"],))
        st.button(t("btn_border"), width="stretch", on_click=apply_preset, args=(SAMPLE_PROFILES["Borderline"],))
        st.button(t("btn_high"), width="stretch", on_click=apply_preset, args=(SAMPLE_PROFILES["High risk"],))

        st.markdown(f'<div class="section-title" style="margin-top:1.2rem"><span class="bar"></span>{t("section_thresholds")}</div>', unsafe_allow_html=True)
        st.slider(t("slider_low"), 0.05, 0.90, key="th_low", step=0.01)
        st.slider(t("slider_high"), 0.10, 0.95, key="th_high", step=0.01)

        st.markdown(f'<div class="section-title" style="margin-top:1.2rem"><span class="bar"></span>{t("section_champion")}</div>', unsafe_allow_html=True)
        st.caption(t("champion_caption"))
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", MODEL_METRICS["roc_auc"])
        m2.metric(t("precision"), MODEL_METRICS["precision"])
        m3, m4 = st.columns(2)
        m3.metric(t("recall"), MODEL_METRICS["recall"])
        m4.metric(t("accuracy"), MODEL_METRICS["accuracy"])
        st.caption(t("champion_note"))


def main():
    st.set_page_config(
        page_title="PayLater Credit Scoring",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    for key, value in APP_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    render_sidebar()

    st.markdown(
        f"""
        <div class="topbar">
            <h1>💳 {t('brand_title')}</h1>
            <p>{t('brand_sub')}</p>
            <div class="status"><span class="dot"></span>{t('status_loaded')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, scaler = get_artifacts()
    except Exception as error:
        st.error(t("load_error", error=error))
        st.stop()

    low, high = st.session_state["th_low"], st.session_state["th_high"]
    if low >= high:
        st.warning(t("threshold_warning"))
        high = min(0.95, low + 0.05)

    st.markdown(f'<div class="section-title" style="margin-top:1.4rem"><span class="bar"></span>{t("section_profile")}</div>', unsafe_allow_html=True)

    with st.form("applicant_form", border=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.number_input(t("field_income"), min_value=1000, max_value=2_000_000,
                            step=1000, key="in_income")
            st.number_input(t("field_loan"), min_value=500, max_value=50_000,
                            step=500, key="in_amnt")
            st.number_input(t("field_age"), min_value=18, max_value=100, step=1, key="in_age")
        with col2:
            st.number_input(t("field_rate"), min_value=1.0, max_value=40.0,
                            step=0.01, format="%.2f", key="in_rate")
            st.number_input(t("field_emp"), min_value=0.0, max_value=60.0,
                            step=0.5, key="in_emp")
            st.selectbox(t("field_home"), HOME_CODES, format_func=lambda c: label_for(HOME_LABELS, c), key="in_home")

        st.markdown(f'<div class="section-title" style="margin-top:0.6rem"><span class="bar"></span>{t("section_intent")}</div>', unsafe_allow_html=True)
        st.radio(t("field_intent"), INTENT_CODES, horizontal=True,
                 format_func=lambda c: label_for(INTENT_LABELS, c),
                 label_visibility="collapsed", key="in_intent")

        with st.expander(t("expander_advanced")):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                st.selectbox(t("field_grade"), GRADE_CODES, key="in_grade")
            with adv2:
                st.selectbox(t("field_default"), DEFAULT_CODES, format_func=lambda c: label_for(DEFAULT_LABELS, c), key="in_default")
            with adv3:
                st.number_input(t("field_hist"), min_value=0, max_value=60, step=1, key="in_hist")

        submitted = st.form_submit_button(t("submit"), width="stretch")

    income = float(st.session_state["in_income"])
    amount = float(st.session_state["in_amnt"])

    st.markdown(f'<div class="section-title" style="margin-top:1.4rem"><span class="bar"></span>{t("section_eval")}</div>', unsafe_allow_html=True)

    if not submitted:
        st.info(t("info_prompt"))
        st.stop()

    result = score_applicant(
        model,
        scaler,
        age=st.session_state["in_age"],
        income=income,
        emp=st.session_state["in_emp"],
        amnt=amount,
        rate=st.session_state["in_rate"],
        hist=st.session_state["in_hist"],
        home=st.session_state["in_home"],
        intent=st.session_state["in_intent"],
        grade=st.session_state["in_grade"],
        default=st.session_state["in_default"],
        low=low,
        high=high,
    )

    probability = result["probability"]
    contributions = result["contributions"]
    decision = result["decision"]
    ratio = result["ratio"]

    stroke_color = {"approve": "#15803d", "review": "#b45309", "reject": "#b91c1c"}[decision]

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        render_decision(probability, decision, ratio, low, high)
    with right:
        st.markdown(
            f'<div class="metric-card">{render_gauge(probability, low, high, stroke_color)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-title" style="margin-top:1.4rem"><span class="bar"></span>{t("section_context")}</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (t("card_exposure"), f"{ratio * 100:.2f}%", t("card_exposure_sub")),
        (t("card_segment"), "LightGBM", t("card_segment_sub")),
        (t("card_vector"), str(len(FEATURE_ORDER)), t("card_vector_sub")),
        (t("card_scaling"), "StandardScaler", t("card_scaling_sub")),
    ]
    for column, (key, value, sub) in zip((c1, c2, c3, c4), cards):
        column.markdown(
            f'<div class="metric-card"><div class="k">{key}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander(t("expander_why"), expanded=True):
        render_contributions(model, contributions)

    with st.expander(t("expander_how")):
        st.markdown(t("how_body"))

    st.caption(t("footer_1"))
    st.caption(t("footer_2"))


if __name__ == "__main__":
    main()
