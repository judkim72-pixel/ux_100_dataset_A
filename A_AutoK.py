
# A_AutoK.py
# UX 100 Dataset 전용 / 무조작(Zero-Interaction) / 한글 UI 탭 대시보드
# 데이터 소스: 리포지토리 루트 'ux_100_dataset.xlsx' → 'Data' 시트
#
# A1. AI 채택 점수 분포 (AI Adoption Index, 0–5)
# A2. 모델/스택 공개도 (Model/Stack Disclosure)
# A3. 책임/윤리 정책 공개 보유율 (Privacy/AI Ethics Policy)
# A4. DesignOps 단계 vs 보안/데이터/애널리틱스 (레이더)
# A5. AI 채택 × 보안 거버넌스 2×2 버블
#
# 실행:
#   streamlit run A_AutoK.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Matplotlib (headless-safe)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
    _MPL_ERR = None
except Exception as _e:
    _MPL_OK = False
    _MPL_ERR = _e

st.set_page_config(page_title="A 모듈: 한글 고정 대시보드", layout="wide")

EXCEL = Path(__file__).resolve().parent / "ux_100_dataset.xlsx"
SHEET = "Data"

# ==== 유틸 ====
def ensure_mpl():
    if not _MPL_OK:
        st.error(f"Matplotlib 불러오기 실패: {_MPL_ERR} — requirements.txt에 matplotlib를 추가하세요.")
        return False
    return True

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
        # 'Not disclosed' 문구가 들어가면 0
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
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# 필요한 컬럼 (정확 명칭 / 영문은 괄호, 표기 그대로)
COL_COMP = "Company"
COL_AI   = "AI Adoption Index (0–5)"
COL_DISC = "Model/Stack Disclosure"
COL_ETH  = "Privacy/AI Ethics Policy (public Y/N)"
COL_DES  = "GenAI in Design Ops"
COL_ISO  = "ISO/IEC 27001 (Y/N)"
COL_SSD  = "Security Review in SDLC"
# 프록시 성숙도(0~100) 산출용 원천 컬럼
COL_AI_ROLES = "AI Roles Present (count)"
COL_ANALYTICS = "Analytics/Experimentation"
COL_SECSDLC_PTS = "SecSDLC_Pts (0-6)"
COL_ISO_PTS = "ISO27001_Pts (0/8)"

missing = [c for c in [COL_COMP, COL_AI, COL_DISC, COL_ETH, COL_DES, COL_ISOYN, COL_SSDYN, COL_SEC_PTS, COL_SSD_PTS, COL_DS_PRESENT, COL_DS_OPEN, COL_ANALYTICS_PTS] if c not in df.columns]
if missing:
    st.error("필수 컬럼이 엑셀에 없습니다: " + ", ".join(missing))
    st.stop()

# 숫자 변환
ai  = to_num(df[COL_AI])
disc = binarize(df[COL_DISC])
eth  = binarize(df[COL_ETH])
# 보안 성숙도(0~100): ISO27001_Pts(0/8), SecSDLC_Pts(0-6) → 각 0~100 환산, 동가중 평균
iso_pts = to_num(df[COL_SEC_PTS])
ssd_pts = to_num(df[COL_SSD_PTS])
sec_iso100 = (iso_pts / 8.0) * 100.0
sec_ssd100 = (ssd_pts / 6.0) * 100.0
secM = (sec_iso100.fillna(0) + sec_ssd100.fillna(0)) / 2.0
# 데이터 사이언스 성숙도(0~100): AI Roles Present(count) 60% + Open Roles(Data/ML/AI) 40% (각각 데이터 내 최대값 정규화)
roles_present = to_num(df[COL_DS_PRESENT])
roles_open = to_num(df[COL_DS_OPEN])
max_present = roles_present.max(skipna=True) if roles_present.notna().any() else 0
max_open = roles_open.max(skipna=True) if roles_open.notna().any() else 0
present100 = (roles_present / max_present * 100.0) if max_present else roles_present*0
open100 = (roles_open / max_open * 100.0) if max_open else roles_open*0
dsM = present100*0.6 + open100*0.4
# 애널리틱스 성숙도(0~100): Analytics_Pts(0-5) → 0~100 환산
an_pts = to_num(df[COL_ANALYTICS_PTS])
anM = (an_pts / 5.0) * 100.0

# DesignOps 단계 매핑

# 탭 구성
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
    if ensure_mpl() and ai.notna().sum() >= 2:
        med = float(ai.median())
        p90 = float(ai.quantile(0.90))
        q1, q3 = float(ai.quantile(0.25)), float(ai.quantile(0.75))

        col1, col2 = st.columns([2,1])
        with col1:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(ai.dropna().values, bins=10)
            ax.axvline(med, linestyle="--")
            ax.set_title("AI 채택 히스토그램 (Histogram)")
            ax.set_xlabel("점수(Score, 0–5)"); ax.set_ylabel("기업 수(Count)")
            st.pyplot(fig, use_container_width=True)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.boxplot(ai.dropna().values, vert=True, showmeans=True)
            ax2.set_title("박스플롯(Boxplot)")
            st.pyplot(fig2, use_container_width=True)

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
    if ensure_mpl() and disc.notna().sum() >= 1:
        rate = float(disc.mean())
        vals = [rate, 1-rate]
        fig, ax = plt.subplots(figsize=(4,4))
        wedges, _ = ax.pie(vals, startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title("공개율 도넛(Donut)")
        st.pyplot(fig, use_container_width=False)
        st.markdown(f"- **공개율(Disclosure Rate)**: {rate*100:.1f}% (모수 N={int(disc.notna().sum())})")
        st.caption("근거: 각 기업의 공개여부 필드를 1/0으로 표준화하여 평균을 산출했습니다.")
        st.caption("해설: 모델·스택 공개는 규제 산업 및 대기업과의 신뢰 형성에 유리하게 작용할 수 있습니다.")
    else:
        st.warning("공개여부 데이터가 부족합니다.")

# ==== A3 ====
with tabs[2]:
    st.header("A3. 책임/윤리 정책 공개 보유율 (AI Ethics/Privacy Policy, Y/N)")
    st.caption("설명: AI 윤리/프라이버시 정책 공개 보유율을 **AI 채택 수준 3분위(상·중·하)**로 비교합니다.")
    mask = ai.notna() & eth.notna()
    if ensure_mpl() and mask.sum() >= 3:
        ai2, eth2 = ai[mask], eth[mask]
        q33, q66 = tertile_bounds(ai2)
        def bucket(v):
            if v < q33: return "하(Low)"
            if v < q66: return "중(Mid)"
            return "상(Top)"
        grp = ai2.map(bucket)
        rates = eth2.groupby(grp).mean().reindex(["하(Low)","중(Mid)","상(Top)"])

        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(len(rates))
        ax.bar(x, rates.values*100)
        ax.set_xticks(x); ax.set_xticklabels(rates.index)
        ax.set_ylabel("정책 공개율 (%)")
        ax.set_title("윤리/프라이버시 정책 공개율 – 채택 수준별")
        st.pyplot(fig, use_container_width=True)

        st.markdown("""""- **근거(Evidence)**: 각 기업의 정책 공개(Y/N)를 1/0으로 정규화 후, 채택 점수 3분위 그룹 평균을 비교했습니다.
- **해설(Commentary)**: 상위 채택군이 더 높은 공개율을 보인다면, 조직적 거버넌스가 대규모 AI 도입과 동행한다는 가설을 지지합니다.""""")
    else:
        st.warning("AI 채택/윤리정책 데이터가 부족합니다. (필요: 최소 3개 유효쌍)")

# ==== A4 ====
with tabs[3]:
    st.header("A4. GenAI in DesignOps 단계 vs 보안/데이터/애널리틱스 (레이더)")
    st.caption("설명: 채택 수준 3분위(상·중·하)의 **DesignOps 단계(0~4→0~100)**와 **보안·데이터·애널리틱스(0~100)** 평균 프로파일을 비교합니다.")
    mask = ai.notna() & des100.notna() & secM.notna() & (dsM.notna() if isinstance(dsM, pd.Series) else False) & (anM.notna() if isinstance(anM, pd.Series) else False)
    if ensure_mpl() and mask.sum() >= 3:
        q33, q66 = tertile_bounds(ai[mask])
        def bucket(v):
            if v < q33: return "하(Low)"
            if v < q66: return "중(Mid)"
            return "상(Top)"
        grp = ai[mask].map(bucket)

        metrics = ["DesignOps(단계)", "Security(보안)", "DataScience(데이터)", "Analytics(애널리틱스)"]
        prof = {}
        for lvl in ["하(Low)","중(Mid)","상(Top)"]:
            m = grp == lvl
            if m.any():
                prof[lvl] = [
                    float(des100[mask][m].mean()),
                    float(secM[mask][m].mean()),
                    float(dsM[mask][m].mean()),
                    float(anM[mask][m].mean()),
                ]

        if prof:
            import numpy as _np
            angles = _np.linspace(0, 2*_np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            fig = plt.figure(figsize=(6,6))
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(_np.pi/2); ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=10)
            ax.set_rlabel_position(0); ax.set_yticks([20,40,60,80]); ax.set_ylim(0,100)

            for lvl, vals in prof.items():
                v = vals + vals[:1]
                ax.plot(angles, v, linewidth=2, label=lvl)
                ax.fill(angles, v, alpha=0.08)

            ax.set_title("그룹 평균 레이더(Radar)")
            ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05))
            st.pyplot(fig, use_container_width=False)

            st.markdown("""- **근거**: DesignOps 단계(0~4)를 0~100으로 스케일링, 나머지는 0~100 지표 그대로 평균.""")
            st.caption("해설: 상위 채택군이 전반 지표에서 우수하다면, 디자인 파이프라인의 자동화·협업·보안 체계가 성숙함을 시사합니다.")
        else:
            st.warning("레이더를 그릴 그룹 데이터가 충분치 않습니다.")
    else:
        st.warning("필요 데이터가 부족합니다. (DesignOps/보안/데이터/애널리틱스 프록시 컬럼을 확인하세요)")

# ==== A5 ====
with tabs[4]:
    st.header("A5. AI 채택 × 보안 거버넌스 2×2 매트릭스 (버블)")
    st.caption("설명: X=AI 채택(0–5→0–100), Y=보안 거버넌스(ISO 27001, SecSDLC 평균×100). **중앙값 기준 2×2 분할**로 위험 구역을 식별합니다.")
    if ensure_mpl():
        ai100 = (ai / (ai.max(skipna=True) if ai.max(skipna=True) else 5)) * 100.0
        gov100 = secM
        m = ai100.notna() & gov100.notna()
        if m.sum() >= 2:
            x, y = ai100[m], gov100[m]
            x_med, y_med = float(x.median()), float(y.median())

            fig, ax = plt.subplots(figsize=(7,5))
            ax.scatter(x, y, s=60, alpha=0.6)
            ax.axvline(x_med, linestyle="--")
            ax.axhline(y_med, linestyle="--")
            ax.set_xlabel("AI 채택 (AI Adoption, 0–100)")
            ax.set_ylabel("보안 거버넌스 (Security Governance, 0–100)")
            ax.set_title("2×2 매트릭스")
            st.pyplot(fig, use_container_width=False)

            risk = (x >= x_med) & (y < y_med)
            st.markdown(f"- **위험 구역**(고채택·저보안): {int(risk.sum())}/{len(x)}개 기업")
            st.caption("해설: 빠른 실험에 비해 보안 체계가 부족한 기업군은 단기간 성과 후 규제·사고 리스크가 확대될 수 있습니다.")
        else:
            st.warning("2×2 매트릭스를 그릴 데이터가 부족합니다.")
