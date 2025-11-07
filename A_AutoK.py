
# A_AutoK.py
# UX 100 Dataset – 한글 탭형 고정 대시보드 (무조작, 인터랙티브 툴팁 지원)
# 데이터: repo 루트 'ux_100_dataset.xlsx' → 'Data' 시트

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Plotly for interactivity (hover tooltips)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception as _e_plotly:
    _PLOTLY_OK = False
    _PLOTLY_ERR = _e_plotly

# Matplotlib only for server-side font registration (optional fallback)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _MPL_OK = True
    _MPL_ERR = None
except Exception as _e:
    _MPL_OK = False
    _MPL_ERR = _e

st.set_page_config(page_title="A 모듈: UX 100 대시보드(한글, 인터랙티브)", layout="wide")

EXCEL = Path(__file__).resolve().parent / "ux_100_dataset.xlsx"
SHEET = "Data"

# ==== 컬럼 상수(엑셀 Data 시트 실제 명칭) ====
COL_COMP = "Company"
COL_AI   = "AI Adoption Index (0–5)"
COL_DISC = "Model/Stack Disclosure"
COL_ETH  = "Privacy/AI Ethics Policy (public Y/N)"
COL_DES  = "GenAI in Design Ops"

# 보안/데이터/애널리틱스 프록시 관련
COL_ISOYN = "ISO/IEC 27001 (Y/N)"
COL_SSDYN = "Security Review in SDLC"
COL_SEC_PTS = "ISO27001_Pts (0/8)"
COL_SSD_PTS = "SecSDLC_Pts (0-6)"
COL_DS_PRESENT = "AI Roles Present (count)"
COL_DS_OPEN = "Open Roles (Data/ML/AI)"
COL_ANALYTICS_PTS = "Analytics_Pts (0-5)"

# ==== 유틸 ====
def ensure_korean_font():
    """Try to prefer 'Malgun Gothic'. If not available on the server,
    we still set Plotly layout fonts (it will render on client if available).
    For Matplotlib fallback we warn if not present."""
    font_name = "Malgun Gothic"
    ok = False
    if _MPL_OK:
        try:
            # Try to find installed font
            path = fm.findfont(font_name, fallback_to_default=False)
            if path and path.lower().endswith(('.ttf', '.otf')):
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                ok = True
        except Exception:
            ok = False
    # Plotly will try to use client font; still set default family
    return font_name, ok

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def binarize(series):
    # 공개/비공개, Y/N, 1/0 등 다양한 표기를 1/0으로 통일
    truthy = {"y","yes","true","t","1","공개","있음","유","disclosed","open","public"}
    falsy  = {"n","no","false","f","0","비공개","없음","무","not disclosed","private"}
    def f(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int,float)) and not pd.isna(x):
            return 1 if x != 0 else 0
        xs = str(x).strip().lower()
        if xs in truthy: return 1
        if xs in falsy:  return 0
        return 0 if "not disclosed" in xs else 1
    return series.map(f)

def tertile_bounds(s: pd.Series):
    s = s.dropna()
    if len(s) < 3:
        return (np.nan, np.nan)
    return float(s.quantile(0.33)), float(s.quantile(0.66))

@st.cache_data(show_spinner=False)
def load_data():
    if not EXCEL.exists():
        raise FileNotFoundError("루트에 'ux_100_dataset.xlsx'가 없습니다.")
    df = pd.read_excel(EXCEL, sheet_name=SHEET, engine="openpyxl")
    return df

# ==== 데이터 로드 ====
try:
    df = load_data()
    # Plotly presence check
    if not _PLOTLY_OK:
        st.error(f"Plotly가 설치되어 있지 않습니다: {_PLOTLY_ERR}. requirements.txt에 plotly를 포함하고 재배포하세요.")
        st.stop()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# 폰트 설정
font_family, font_ok = ensure_korean_font()
if not font_ok:
    st.info("서버에 '맑은 고딕(Malgun Gothic)' 폰트가 없어도 브라우저에서 정상 표시될 수 있습니다. "
            "서버 렌더링 폰트가 필요하면 리포지토리에 TTF를 포함해 주세요.")

# Plotly 기본 폰트 적용
px.defaults.template = "plotly_white"
px.defaults.width = None
px.defaults.height = None

# 필수 컬럼 체크
missing = [c for c in [COL_COMP, COL_AI, COL_DISC, COL_ETH, COL_DES,
                       COL_ISOYN, COL_SSDYN, COL_SEC_PTS, COL_SSD_PTS,
                       COL_DS_PRESENT, COL_DS_OPEN, COL_ANALYTICS_PTS] if c not in df.columns]
if missing:
    st.error("필수 컬럼이 엑셀에 없습니다: " + ", ".join(missing))
    st.stop()

# ---- 수치/지표 계산 (탭 렌더 전, 한 번만) ----
ai   = to_num(df[COL_AI])
disc = binarize(df[COL_DISC])
eth  = binarize(df[COL_ETH])

# DesignOps 단계 → 0~4 → 0~100
_stage_map = {
    "none":0, "asset":1, "assist":2, "cocreate":3, "co-create":3, "co create":3, "e2e":4,
    "0":0, "1":1, "2":2, "3":3, "4":4
}
def _stage(x):
    if pd.isna(x): return np.nan
    xs = str(x).strip().lower()
    return _stage_map.get(xs, pd.to_numeric(xs, errors="coerce"))
des_raw = df[COL_DES].map(_stage)
des100  = (des_raw / 4.0) * 100.0

# 보안 성숙도(0~100): ISO27001_Pts(0/8), SecSDLC_Pts(0-6) → 각 0~100 환산, 동가중 평균
iso_pts = to_num(df[COL_SEC_PTS])
ssd_pts = to_num(df[COL_SSD_PTS])
sec_iso100 = (iso_pts / 8.0) * 100.0
sec_ssd100 = (ssd_pts / 6.0) * 100.0
secM = (sec_iso100.fillna(0) + sec_ssd100.fillna(0)) / 2.0

# 데이터 사이언스 성숙도(0~100): AI Roles Present(count) 60% + Open Roles 40% (최대값 정규화)
roles_present = to_num(df[COL_DS_PRESENT])
roles_open    = to_num(df[COL_DS_OPEN])
max_present = roles_present.max(skipna=True) if roles_present.notna().any() else 0
max_open    = roles_open.max(skipna=True) if roles_open.notna().any() else 0
present100  = (roles_present / max_present * 100.0) if max_present else roles_present * 0
open100     = (roles_open / max_open * 100.0) if max_open else roles_open * 0
dsM = present100 * 0.6 + open100 * 0.4

# 애널리틱스 성숙도(0~100): Analytics_Pts(0-5) → 0~100
an_pts = to_num(df[COL_ANALYTICS_PTS])
anM    = (an_pts / 5.0) * 100.0

# ---- 탭 구성 ----
tabs = st.tabs([
    "A1. AI 채택 분포 (AI Adoption Index)",
    "A2. 모델/스택 공개도 (Disclosure Rate)",
    "A3. 윤리 정책 보유율 (AI Ethics Policy)",
    "A4. DesignOps × 보안/데이터/애널리틱스 (레이더)",
    "A5. AI 채택 × 보안 거버넌스 (2×2)",
])

# ==== A1 ====
with tabs[0]:
    st.header("A1. AI 채택 점수 분포 (AI Adoption Index)")
    st.caption("설명(Explanation): 업계 내 AI 채택 수준(0–5 스케일)의 전반적 분포를 확인합니다.")
    valid = ai.dropna()
    if len(valid) >= 2:
        med = float(valid.median())
        p90 = float(valid.quantile(0.90))
        q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))

        # Histogram
        fig = px.histogram(pd.DataFrame({"AI 채택 점수": valid}), x="AI 채택 점수", nbins=10,
                           title="AI 채택 히스토그램 (Histogram)")
        fig.update_layout(font_family=font_family)
        fig.add_vline(x=med, line_dash="dash", annotation_text=f"중앙값 {med:.2f}", annotation_position="top")
        fig.update_traces(hovertemplate="점수: %{x}<br>기업 수: %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        # Box
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=valid, boxmean="sd", name="분포"))
        fig2.update_layout(title="박스플롯(Boxplot)", font_family=font_family, showlegend=False, yaxis_title="점수(0–5)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""- **중앙값(Median)**: {med:.2f}
- **상위 10% 컷오프(Top 10% Cutoff)**: {p90:.2f}
- **IQR(사분위 범위)**: {q1:.2f}–{q3:.2f}""")
        st.caption("해설: 중앙값과 IQR을 통해 대부분 기업의 채택 구간을 파악하고, 상위 10% 컷오프로 선도군을 식별할 수 있습니다.")
    else:
        st.warning("AI 채택 데이터가 부족합니다. (필요: 최소 2개 유효값)")

# ==== A2 ====
with tabs[1]:
    st.header("A2. 모델/스택 공개도 (Model/Stack Disclosure Rate)")
    st.caption("설명: 모델 또는 기술 스택을 **공개**하는 기업 비율을 확인합니다.")
    valid = disc.dropna()
    if len(valid) >= 1:
        rate = float(valid.mean())
        fig = go.Figure(data=[go.Pie(labels=["공개", "비공개"], values=[rate, 1-rate], hole=0.4,
                                     hovertemplate="%{label}: %{percent:.1%}<extra></extra>")])
        fig.update_layout(title="공개율 도넛(Donut)", font_family=font_family)
        st.plotly_chart(fig, use_container_width=False)
        st.markdown(f"- **공개율(Disclosure Rate)**: {rate*100:.1f}% (모수 N={int(len(valid))})")
        st.caption("근거: 각 기업의 공개여부 필드를 1/0으로 표준화하여 평균을 산출했습니다.")
        st.caption("해설: 모델·스택 공개는 규제 산업 및 대기업과의 신뢰 형성에 유리하게 작용할 수 있습니다.")
    else:
        st.warning("공개여부 데이터가 부족합니다.")

# ==== A3 ====
with tabs[2]:
    st.header("A3. 책임/윤리 정책 공개 보유율 (AI Ethics/Privacy Policy, Y/N)")
    st.caption("설명: AI 윤리/프라이버시 정책 공개 보유율을 **AI 채택 수준 3분위(상·중·하)**로 비교합니다.")
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

        fig = px.bar(x=rates.index, y=(rates.values*100),
                     labels={"x":"채택 그룹", "y":"정책 공개율(%)"},
                     title="윤리/프라이버시 정책 공개율 – 채택 수준별")
        fig.update_layout(font_family=font_family)
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""- **근거(Evidence)**: 각 기업의 정책 공개(Y/N)를 1/0으로 정규화 후, 채택 점수 3분위 그룹 평균을 비교했습니다.
- **해설(Commentary)**: 상위 채택군이 더 높은 공개율을 보인다면, 조직적 거버넌스가 대규모 AI 도입과 동행한다는 가설을 지지합니다.""")
    else:
        st.warning("AI 채택/윤리정책 데이터가 부족합니다. (필요: 최소 3개 유효쌍)")

# ==== A4 ====
with tabs[3]:
    st.header("A4. GenAI in DesignOps 단계 vs 보안/데이터/애널리틱스 (레이더)")
    st.caption("설명: 채택 수준 3분위(상·중·하)의 **DesignOps 단계(0~4→0~100)**와 **보안·데이터·애널리틱스(0~100)** 평균 프로파일을 비교합니다.")
    # 보다 관대한 방식: 그룹은 AI 채택 값이 있는 행으로만 구성하고,
    # 각 지표(DesignOps, Security, DataScience, Analytics)는 가능한 값만 평균합니다.
    valid_ai = ai.dropna()
    if len(valid_ai) >= 3:
        # tertile 경계
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

        # 그룹별 평균(사용 가능한 값만)
        prof = {"하(Low)": [], "중(Mid)": [], "상(Top)": []}
        kept_metrics = []
        counts = {g: [] for g in prof.keys()}

        for label, series in metrics:
            any_finite = False
            group_means = {}
            group_counts = {}
            for g in ["하(Low)", "중(Mid)", "상(Top)"]:
                m = (grp_all == g) & series.notna()
                n = int(m.sum())
                v = float(series[m].mean()) if n > 0 else np.nan
                group_means[g] = v
                group_counts[g] = n
                if n > 0 and np.isfinite(v):
                    any_finite = True
            if any_finite:
                kept_metrics.append(label)
                for g in ["하(Low)", "중(Mid)", "상(Top)"]:
                    prof[g].append(group_means[g] if np.isfinite(group_means[g]) else None)
                    counts[g].append(group_counts[g])

        if not kept_metrics:
            st.warning("A4 레이더를 그릴 수 있는 유효 지표가 없습니다. (해당 지표들의 결측률이 매우 높음)")
        else:
            fig = go.Figure()
            for g in ["하(Low)", "중(Mid)", "상(Top)"]:
                vals = prof[g]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=kept_metrics, name=g, fill="toself",
                    connectgaps=False, hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                showlegend=True,
                title="그룹 평균 레이더(Radar)",
                font_family=font_family
            )
            st.plotly_chart(fig, use_container_width=False)

            # 보조 표: 각 축별 유효 표본수(N)
            import pandas as _pd
            n_table = _pd.DataFrame({
                "Metric": kept_metrics,
                "하(Low) N": counts["하(Low)"],
                "중(Mid) N": counts["중(Mid)"],
                "상(Top) N": counts["상(Top)"],
            })
            st.dataframe(n_table, use_container_width=True)

            st.markdown("- **근거**: DesignOps 단계(0~4)→0~100 스케일링. 보안=ISO/SecSDLC 점수 환산(0~100) 평균, 데이터=Roles Present/Open 정규화 가중합, 애널리틱스=Analytics_Pts 환산(0~100).")
            st.caption("해설: 일부 지표가 결측이어도 사용 가능한 값만으로 평균을 산출합니다. 표본수가 적은 축은 하단 표에서 N을 확인하세요.")
    else:
        st.warning("AI 채택 값이 3개 미만입니다. (A4 레이더 계산 불가)")

# ==== A5 ====

with tabs[4]:
    st.header("A5. AI 채택 × 보안 거버넌스 2×2 매트릭스 (버블)")
    st.caption("설명: X=AI 채택(0–5→0–100), Y=보안 거버넌스(ISO27001_Pts & SecSDLC_Pts → 0~100 평균). **중앙값 기준 2×2 분할**로 위험 구역을 식별합니다.")
    # 스케일 정규화
    ai_max = ai.max(skipna=True) if ai.notna().any() else 5
    ai100 = (ai / (ai_max if ai_max else 5)) * 100.0
    gov100 = secM  # 이미 0~100

    m = ai100.notna() & gov100.notna()
    if m.sum() >= 2:
        x, y = ai100[m], gov100[m]
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

        risk = (x >= x_med) & (y < y_med)
        st.markdown(f"- **위험 구역**(고채택·저보안): {int(risk.sum())}/{len(x)}개 기업")
        st.caption("해설: 빠른 실험에 비해 보안 체계가 부족한 기업군은 단기간 성과 후 규제·사고 리스크가 확대될 수 있습니다.")
    else:
        st.warning("2×2 매트릭스를 그릴 데이터가 부족합니다.")
