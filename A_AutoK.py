# A_AutoK.py — UX 100 Dashboard (KR, Plotly, Tooltips)
# A1: Histogram + ECDF
# A2: Donut + disclosed company table
# A3: Bar with 95% CI
# A4: Radar with tolerant grouping + coverage banner + commentary
# A5: 2x2 scatter with coverage banner + light jitter if y≈0 + commentary

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Plotly (guard)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception as _e:
    _PLOTLY_OK = False
    _PLOTLY_ERR = _e

st.set_page_config(page_title="A 모듈: UX 100 대시보드(한글, 인터랙티브)", layout="wide")

EXCEL = Path(__file__).resolve().parent / "ux_100_dataset.xlsx"
SHEET = "Data"

# ==== Columns ====
COL_COMP = "Company"
COL_AI   = "AI Adoption Index (0–5)"
COL_DISC = "Model/Stack Disclosure"
COL_ETH  = "Privacy/AI Ethics Policy (public Y/N)"
COL_DES  = "GenAI in Design Ops"
COL_ISOYN = "ISO/IEC 27001 (Y/N)"
COL_SSDYN = "Security Review in SDLC"
COL_SEC_PTS = "ISO27001_Pts (0/8)"
COL_SSD_PTS = "SecSDLC_Pts (0-6)"
COL_DS_PRESENT = "AI Roles Present (count)"
COL_DS_OPEN = "Open Roles (Data/ML/AI)"
COL_ANALYTICS_PTS = "Analytics_Pts (0-5)"

def to_num(s): 
    return pd.to_numeric(s, errors="coerce")

def binarize(series):
    truthy = {"y","yes","true","t","1","공개","있음","유","disclosed","open","public"}
    falsy  = {"n","no","false","f","0","비공개","없음","무","not disclosed","private"}
    def f(x):
        if pd.isna(x): return np.nan
        if isinstance(x,(int,float)) and not pd.isna(x): return 1 if x!=0 else 0
        xs = str(x).strip().lower()
        if xs in truthy: return 1
        if xs in falsy: return 0
        return 0 if "not disclosed" in xs else 1
    return series.map(f)

def tertile_bounds(s: pd.Series):
    s = s.dropna()
    if len(s) < 3: return (np.nan, np.nan)
    return float(s.quantile(0.33)), float(s.quantile(0.66))

def ci95(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    n = len(x)
    if n < 2: return (np.nan, np.nan)
    m = float(x.mean()); sd = float(x.std(ddof=1))
    hw = 1.96 * sd / np.sqrt(n)
    return (m - hw, m + hw)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel(EXCEL, sheet_name=SHEET, engine="openpyxl")
    return df

# ==== Load ====
if not _PLOTLY_OK:
    st.error(f"Plotly가 설치되어 있지 않습니다: {_PLOTLY_ERR}. requirements.txt를 설치하세요.")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# Column check
missing = [c for c in [COL_COMP, COL_AI, COL_DISC, COL_ETH, COL_DES,
                       COL_ISOYN, COL_SSDYN, COL_SEC_PTS, COL_SSD_PTS,
                       COL_DS_PRESENT, COL_DS_OPEN, COL_ANALYTICS_PTS] if c not in df.columns]
if missing:
    st.error("필수 컬럼이 엑셀에 없습니다: " + ", ".join(missing))
    st.stop()

# Defaults
px.defaults.template = "plotly_white"
font_family = "Malgun Gothic"

# ==== Metrics ====
ai   = to_num(df[COL_AI])
disc = binarize(df[COL_DISC])
eth  = binarize(df[COL_ETH])

# DesignOps stage (KR variants)
_stage_map = {
    "none":0, "없음":0, "0":0,
    "asset":1, "에셋":1, "자산":1, "1":1,
    "assist":2, "보조":2, "지원":2, "2":2,
    "cocreate":3, "co-create":3, "co create":3, "co-create":3, "공동":3, "공동창작":3, "협업":3, "3":3,
    "e2e":4, "endtoend":4, "end-to-end":4, "엔드투엔드":4, "4":4
}
def _stage(x):
    if pd.isna(x): return np.nan
    xs = str(x).strip().lower()
    return _stage_map.get(xs, pd.to_numeric(xs, errors="coerce"))
des_raw = df[COL_DES].map(_stage)
des100  = (des_raw / 4.0) * 100.0

# Security maturity (row-wise mean of available: points first, else Y/N fallback)
iso_pts = to_num(df[COL_SEC_PTS])
ssd_pts = to_num(df[COL_SSD_PTS])
sec_iso100_pts = (iso_pts / 8.0) * 100.0
sec_ssd100_pts = (ssd_pts / 6.0) * 100.0
sec_iso100_yn  = binarize(df[COL_ISOYN]) * 100.0
sec_ssd100_yn  = binarize(df[COL_SSDYN]) * 100.0
sec_stack = pd.concat([sec_iso100_pts, sec_ssd100_pts, sec_iso100_yn, sec_ssd100_yn], axis=1)
secM = sec_stack.mean(axis=1, skipna=True)

# DataScience maturity (present/open max-normalized, 60/40)
roles_present = to_num(df[COL_DS_PRESENT])
roles_open    = to_num(df[COL_DS_OPEN])
max_present = roles_present.max(skipna=True) if roles_present.notna().any() else 0
max_open    = roles_open.max(skipna=True) if roles_open.notna().any() else 0
present100  = (roles_present / max_present * 100.0) if max_present else roles_present*0
open100     = (roles_open / max_open * 100.0) if max_open else roles_open*0
dsM = present100 * 0.6 + open100 * 0.4

# Analytics maturity (points; if all NaN, fall back to Y/N column)
an_pts = to_num(df[COL_ANALYTICS_PTS])
anM    = (an_pts / 5.0) * 100.0
if anM.isna().all() and 'Analytics/Experimentation' in df.columns:
    anM = binarize(df['Analytics/Experimentation']) * 100.0

# ==== Tabs ====
tabs = st.tabs([
    "A1. 채택 분포",
    "A2. 공개도",
    "A3. 윤리정책",
    "A4. DesignOps vs 보안/데이터/애널리틱스",
    "A5. 채택 × 보안 2×2",
])

# === A1 ===
with tabs[0]:
    st.header("A1. AI 채택 점수 분포 (Histogram + ECDF)")
    valid = ai.dropna()
    if len(valid) >= 2:
        med = float(valid.median())
        p90 = float(valid.quantile(0.90))
        q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))

        # Histogram
        fig = px.histogram(pd.DataFrame({"AI 채택 점수": valid}), x="AI 채택 점수", nbins=10,
                           title="히스토그램 (Histogram)")
        fig.update_layout(font_family=font_family)
        fig.add_vline(x=med, line_dash="dash", annotation_text=f"중앙값 {med:.2f}")
        fig.update_traces(hovertemplate="점수: %{x}<br>기업 수: %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        # ECDF
        arr = np.sort(valid.values)
        y = np.arange(1, len(arr)+1) / len(arr)
        fig2 = go.Figure(go.Scatter(x=arr, y=y, mode="lines"))
        fig2.update_layout(title="누적분포(ECDF)", xaxis_title="점수(0–5)", yaxis_title="누적비율",
                           font_family=font_family)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""- **중앙값(Median)**: {med:.2f}
- **상위 10% 컷오프(Top 10% Cutoff)**: {p90:.2f}
- **IQR(사분위 범위)**: {q1:.2f}–{q3:.2f}""")
        st.caption("해설: ECDF는 상·하위 백분위 위치와 꼬리 분포를 직관적으로 보여줍니다.")
    else:
        st.info("AI 채택 데이터가 부족합니다. (필요: 최소 2개 유효값)")

# === A2 ===
with tabs[1]:
    st.header("A2. 모델/스택 공개도 (Donut + 공개기업 표)")
    valid = disc.dropna()
    if len(valid) >= 1:
        rate = float(valid.mean())
        fig = go.Figure(data=[go.Pie(labels=["공개", "비공개"], values=[rate, 1-rate], hole=0.45,
                                     hovertemplate="%{label}: %{percent:.1%}<extra></extra>")])
        fig.update_layout(title="공개율 도넛(Donut)", font_family=font_family)
        st.plotly_chart(fig, use_container_width=False)
        st.markdown(f"- **공개율(Disclosure Rate)**: {rate*100:.1f}% (모수 N={int(len(valid))})")
        st.caption("근거: 공개 여부를 1/0으로 표준화하여 평균을 산출.")

        # 공개 기업 표 (상위 100행)
        disclosed_mask = disc == 1
        disclosed_df = df.loc[disclosed_mask, [COL_COMP, COL_DISC]].copy()
        if len(disclosed_df) == 0:
            st.info("공개 기업이 없습니다.")
        else:
            st.caption("공개 기업(최대 100개)")
            st.dataframe(disclosed_df.head(100), use_container_width=True)
    else:
        st.info("공개여부 데이터가 부족합니다.")

# === A3 ===
with tabs[2]:
    st.header("A3. 윤리/프라이버시 정책 공개율 (Bar + 95% CI)")
    mask = ai.notna() & eth.notna()
    if mask.sum() >= 3:
        ai2, eth2 = ai[mask], eth[mask]
        q33, q66 = tertile_bounds(ai2)
        def bucket(v):
            if v < q33: return "하(Low)"
            if v < q66: return "중(Mid)"
            return "상(Top)"
        grp = ai2.map(bucket)
        rates = eth2.groupby(grp).mean().reindex(["하(Low)","중(Mid)","상(Top)"])
        # 95% CI (상한만 error bar로 표시)
        def _err_plus(g):
            lo, hi = ci95(eth2[grp==g])
            if np.isnan(hi): return 0
            return (hi - rates.loc[g]) * 100
        err_plus = [_err_plus(g) for g in rates.index]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=rates.index, y=rates.values*100,
                             error_y=dict(type="data", array=err_plus)))
        fig.update_layout(title="정책 공개율 + 95% CI", yaxis_title="%", font_family=font_family)
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""- **근거(Evidence)**: 공개(Y/N)를 1/0으로 정규화 후, 채택 점수 3분위 그룹 평균과 95% 신뢰구간을 표기.
- **해설(Commentary)**: 상위 채택군에서 공개율이 높다면 거버넌스 정착과 대규모 도입 간의 연계를 시사합니다.""")
    else:
        st.info("AI 채택/윤리정책 데이터가 부족합니다. (필요: 최소 3개 유효쌍)")

# === A4 ===
with tabs[3]:
    st.header("A4. GenAI in DesignOps 단계 vs 보안/데이터/애널리틱스 (레이더)")
    st.caption(f"데이터 커버리지 – DesignOps:{des100.notna().mean()*100:.1f}% / 보안:{secM.notna().mean()*100:.1f}% / 데이터:{dsM.notna().mean()*100:.1f}% / 애널리틱스:{anM.notna().mean()*100:.1f}%")
    valid_ai = ai.dropna()
    if len(valid_ai) >= 3:
        q33, q66 = tertile_bounds(valid_ai)
        def bucket(v):
            if v < q33: return "하(Low)"
            if v < q66: return "중(Mid)"
            return "상(Top)"
        grp_all = ai.map(lambda v: bucket(v) if pd.notna(v) else np.nan)

        metrics = [
            ("DesignOps(단계)", des100),
            ("Security(보안)",  secM),
            ("DataScience(데이터)", dsM),
            ("Analytics(애널리틱스)", anM),
        ]

        prof = {"하(Low)": [], "중(Mid)": [], "상(Top)": []}
        kept = []
        for label, series in metrics:
            any_val = False
            gm = {}
            for g in ["하(Low)","중(Mid)","상(Top)"]:
                m = (grp_all==g) & series.notna()
                v = float(series[m].mean()) if m.sum()>0 else np.nan
                gm[g] = v
                if m.sum()>0 and np.isfinite(v): any_val = True
            if any_val:
                kept.append(label)
                for g in ["하(Low)","중(Mid)","상(Top)"]:
                    prof[g].append(gm[g] if np.isfinite(gm[g]) else None)

        if kept:
            fig = go.Figure()
            for g in ["하(Low)","중(Mid)","상(Top)"]:
                fig.add_trace(go.Scatterpolar(r=prof[g], theta=kept, name=g, fill="toself",
                                              hovertemplate="%{theta}: %{r:.1f}<extra></extra>"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                              showlegend=True, title="그룹 평균 레이더(Radar)",
                              font_family=font_family)
            st.plotly_chart(fig, use_container_width=False)

            # Commentary
            st.markdown("""
**근거(Evidence).** DesignOps 단계(0~4→0~100), 보안(ISO27001 점수·SecSDLC 점수 또는 Y/N 대체)의 행 단위 평균, 
데이터사이언스(채용/공개 포지션의 정규화 가중합), **애널리틱스(Analytics_Pts 0–5 → 0~100; 전부 결측이면 `Analytics/Experimentation` Y/N 대체)** 를 사용했습니다.

**해설(Commentary).** 상·중·하(3분위) 그룹의 평균 프로파일 차이를 통해 
디자인 파이프라인 성숙도와 **보안·데이터·애널리틱스 역량 간의 동행 관계**를 파악합니다. 
축 간 격차가 작고 커버리지가 낮으면 실제 차이가 작거나 데이터가 부족한 상태일 수 있습니다.
""")

            # Analytics coverage / note
            an_cov = float(anM.notna().mean())*100 if len(anM)>0 else 0.0
            if an_cov < 1.0 or anM.dropna().nunique() <= 1:
                st.info("""**애널리틱스 지표 안내** — 현재 'Analytics_Pts (0-5)' 값이 거의 없거나 단일값입니다. 
가능하면 제품 분석/실험(AB, 페널/퍼널, 코호트, 실험 자동화 등)을 0–5 점수로 기록해 주세요. 
점수 전부 결측일 때는 보조로 `Analytics/Experimentation`(Y/N)을 0/100으로 환산해 사용합니다.""")
        else:
            st.warning("A4 레이더를 그릴 수 있는 유효 지표가 없습니다.")
    else:
        st.info("AI 채택 값이 3개 미만입니다.")

# === A5 ===
with tabs[4]:
    st.header("A5. AI 채택 × 보안 거버넌스 2×2 (Scatter)")
    ai_max = ai.max(skipna=True) if ai.notna().any() else 5
    ai100 = (ai / (ai_max if ai_max else 5)) * 100.0
    gov100 = secM.copy()

    m = ai100.notna() & gov100.notna()
    cov_ai = float(ai100.notna().mean())*100 if len(ai100)>0 else 0
    cov_sec= float(gov100.notna().mean())*100 if len(gov100)>0 else 0
    st.caption(f"데이터 커버리지 – AI:{cov_ai:.1f}% / 보안:{cov_sec:.1f}%")

    if m.sum() >= 2:
        x, y = ai100[m], gov100[m]
        # if y collapsed near zero, add tiny jitter for visual separation only
        if y.std(skipna=True) < 1e-6:
            rng = np.random.default_rng(42)
            y = y + rng.normal(0, 0.5, size=len(y))
        x_med, y_med = float(x.median()), float(y.median())

        fig = px.scatter(pd.DataFrame({
            "AI 채택(0~100)": x,
            "보안 거버넌스(0~100)": y,
            "기업": df.loc[m, COL_COMP]
        }), x="AI 채택(0~100)", y="보안 거버넌스(0~100)", hover_name="기업",
           title="2×2 매트릭스")
        fig.add_vline(x=x_med, line_dash="dash", annotation_text=f"X 중앙값 {x_med:.1f}")
        fig.add_hline(y=y_med, line_dash="dash", annotation_text=f"Y 중앙값 {y_med:.1f}")
        fig.update_traces(hovertemplate="기업: %{hovertext}<br>AI: %{x:.1f}<br>보안: %{y:.1f}<extra></extra>")
        fig.update_layout(font_family=font_family, xaxis=dict(range=[0,100]), yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
**해석 가이드.**
- 우상단(**Q1: 고채택·고보안**)은 **대규모 적용 준비도**가 높습니다.
- 좌상단(**Q2: 저채택·고보안**)은 **규제·보안 요건을 갖춘 탐색 단계**로 볼 수 있습니다.
- 좌하단(**Q3: 저채택·저보안**)은 **초기/관망** 구역입니다.
- 우하단(**Q4: 고채택·저보안**)은 **운영·규제 리스크**가 커질 수 있어 **우선 보완 권고** 대상입니다.

**근거(Evidence).** X=AI 채택(0–5를 0~100으로 정규화), Y=보안 거버넌스(ISO/SecSDLC 점수·Y/N 대체의 행 평균). 
점군이 한 축으로 뭉치면(특히 Y축이 0 부근) 데이터 자체의 저분산/결측 영향일 수 있으며, 표시상 구분을 위해 미세한 지터만 시각적으로 적용했습니다(분석에는 영향 없음).
""")
    else:
        st.info("2×2 매트릭스를 그릴 데이터가 부족합니다.")
