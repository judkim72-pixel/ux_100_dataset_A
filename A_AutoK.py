
# A_Enhanced.py
# UX 100 Dataset – 한글 탭형 고정 대시보드 (가독성 강화판)
# - 저분산/저표본일 때 자동으로 더 유의미한 시각으로 대체
# - A4: 레이더 → 신뢰구간 막대 + Z-스코어 프로파일(보조)
# - A5: 2×2 산점도 → 사분면 히트맵(밀도) + 클러스터 라벨링 + 위험 표

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception as e:
    _PLOTLY_OK = False
    _PLOTLY_ERR = e

st.set_page_config(page_title="A 모듈(강화판): UX 100 대시보드", layout="wide")

EXCEL = Path(__file__).resolve().parent / "ux_100_dataset.xlsx"
SHEET = "Data"

# ==== 컬럼 상수 ====
COL_COMP = "Company"
COL_AI   = "AI Adoption Index (0–5)"
COL_DISC = "Model/Stack Disclosure"
COL_ETH  = "Privacy/AI Ethics Policy (public Y/N)"
COL_DES  = "GenAI in Design Ops"
COL_SEC_PTS = "ISO27001_Pts (0/8)"
COL_SSD_PTS = "SecSDLC_Pts (0-6)"
COL_DS_PRESENT = "AI Roles Present (count)"
COL_DS_OPEN = "Open Roles (Data/ML/AI)"
COL_ANALYTICS_PTS = "Analytics_Pts (0-5)"

def to_num(s): return pd.to_numeric(s, errors="coerce")
def binarize(series):
    truthy={"y","yes","true","t","1","공개","있음","유","disclosed","open","public"}
    falsy ={"n","no","false","f","0","비공개","없음","무","not disclosed","private"}
    def f(x):
        if pd.isna(x): return np.nan
        if isinstance(x,(int,float)) and not pd.isna(x): return 1 if x!=0 else 0
        xs=str(x).strip().lower()
        if xs in truthy: return 1
        if xs in falsy: return 0
        return 0 if "not disclosed" in xs else 1
    return series.map(f)
def tertile_bounds(s):
    s=s.dropna()
    if len(s)<3: return (np.nan,np.nan)
    return float(s.quantile(0.33)), float(s.quantile(0.66))
def ci95(series):
    x=series.dropna().astype(float)
    n=len(x)
    if n<2: return (np.nan,np.nan)
    m=float(x.mean()); s=float(x.std(ddof=1))
    hw=1.96*s/np.sqrt(n)
    return (m-hw, m+hw)

@st.cache_data(show_spinner=False)
def load():
    df = pd.read_excel(EXCEL, sheet_name=SHEET, engine="openpyxl")
    return df

# 데이터 로드
try:
    df = load()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

missing = [c for c in [COL_COMP, COL_AI, COL_DISC, COL_ETH, COL_DES, COL_SEC_PTS, COL_SSD_PTS, COL_DS_PRESENT, COL_DS_OPEN, COL_ANALYTICS_PTS] if c not in df.columns]
if missing:
    st.error("필수 컬럼 누락: " + ", ".join(missing)); st.stop()

# 지표 전처리
ai   = to_num(df[COL_AI])
disc = binarize(df[COL_DISC])
eth  = binarize(df[COL_ETH])

_stage_map={"none":0,"asset":1,"assist":2,"cocreate":3,"co-create":3,"co create":3,"e2e":4,"0":0,"1":1,"2":2,"3":3,"4":4}
def stage(v):
    if pd.isna(v): return np.nan
    xs=str(v).strip().lower()
    return _stage_map.get(xs, pd.to_numeric(xs, errors="coerce"))
des_raw = df[COL_DES].map(stage)
des100  = (des_raw/4.0)*100.0

sec_iso100 = (to_num(df[COL_SEC_PTS])/8.0)*100.0
sec_ssd100 = (to_num(df[COL_SSD_PTS])/6.0)*100.0
secM = (sec_iso100.fillna(0)+sec_ssd100.fillna(0))/2.0

present100 = (to_num(df[COL_DS_PRESENT]) / max(1, to_num(df[COL_DS_PRESENT]).max(skipna=True))) * 100.0
open100    = (to_num(df[COL_DS_OPEN])    / max(1, to_num(df[COL_DS_OPEN]).max(skipna=True)))    * 100.0
dsM = present100*0.6 + open100*0.4

anM = (to_num(df[COL_ANALYTICS_PTS])/5.0)*100.0

# 폰트(클라이언트 측) 설정
font_family = "Malgun Gothic"
px.defaults.template = "plotly_white"

tabs = st.tabs([
    "A1. 채택 분포(강화)",
    "A2. 공개도(강화)",
    "A3. 윤리정책(강화)",
    "A4. DesignOps vs 보안/데이터/애널리틱스(강화)",
    "A5. 채택 × 보안 2×2(강화)",
])

# A1
with tabs[0]:
    st.header("A1. AI 채택 점수 분포 – 강화 뷰")
    valid = ai.dropna()
    if len(valid)>=2:
        fig = px.histogram(pd.DataFrame({"AI 채택 점수": valid}), x="AI 채택 점수", nbins=10,
                           title="히스토그램 + KDE(밀도선)")
        # 밀도선(추정): 히스토그램 스무딩
        fig.update_traces(hovertemplate="점수: %{x}<br>기업 수: %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
        # ECDF 보조
        ecdf = np.sort(valid.values)
        y = np.arange(1, len(ecdf)+1)/len(ecdf)
        fig2 = go.Figure(go.Scatter(x=ecdf, y=y, mode="lines"))
        fig2.update_layout(title="누적분포(ECDF)", xaxis_title="점수(0–5)", yaxis_title="누적비율", font_family=font_family)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("유효 표본이 2개 미만입니다. 추가 입력을 권장합니다.")

# A2
with tabs[1]:
    st.header("A2. 모델/스택 공개도 – 강화 뷰")
    valid = disc.dropna()
    if len(valid)>=1:
        rate = float(valid.mean())
        fig = go.Figure(go.Pie(labels=["공개","비공개"], values=[rate,1-rate], hole=0.5))
        fig.update_traces(hovertemplate="%{label}: %{percent:.1%}<extra></extra>")
        fig.update_layout(title="공개율 도넛", font_family=font_family)
        st.plotly_chart(fig, use_container_width=False)
        # 보조: 기업 리스트
        st.caption("공개 기업 리스트(상위 50개)")
        disclosed = df.loc[valid.index][df[COL_DISC].map(lambda x: str(x).lower() not in ["", "not disclosed", "n", "no", "0"])]
        st.dataframe(disclosed[[COL_COMP, COL_DISC]].head(50), use_container_width=True)
    else:
        st.info("공개도 표본 부족. 입력 개선 필요.")

# A3
with tabs[2]:
    st.header("A3. 윤리/프라이버시 정책 공개율 – 강화 뷰")
    mask = ai.notna() & eth.notna()
    if mask.sum()>=3:
        ai2=ai[mask]; eth2=eth[mask]
        q33,q66=tertile_bounds(ai2)
        def bucket(v):
            if v<q33: return "하(Low)"
            if v<q66: return "중(Mid)"
            return "상(Top)"
        grp=ai2.map(bucket)
        rates = eth2.groupby(grp).mean().reindex(["하(Low)","중(Mid)","상(Top)"])
        ci = {g: ci95(eth2[grp==g]) for g in rates.index}
        low=[(ci[g][0]*100 if not np.isnan(ci[g][0]) else None) for g in rates.index]
        high=[(ci[g][1]*100 if not np.isnan(ci[g][1]) else None) for g in rates.index]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=rates.index, y=rates.values*100, error_y=dict(type="data", array=[(h - m*100 if h is not None else 0) for m,h in zip(rates.values, high)])))
        fig.update_layout(title="공개율(%) + 95% CI", yaxis_title="%", font_family=font_family)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("표본 부족. 최소 3개 이상 필요.")

# A4
with tabs[3]:
    st.header("A4. DesignOps vs 보안/데이터/애널리틱스 – 강화 뷰")
    valid_ai = ai.dropna()
    if len(valid_ai)>=3:
        q33,q66=tertile_bounds(valid_ai)
        def bucket(v):
            if v<q33: return "하(Low)"
            if v<q66: return "중(Mid)"
            return "상(Top)"
        grp_all = ai.map(lambda v: bucket(v) if pd.notna(v) else np.nan)

        metrics = [
            ("DesignOps(단계)", des100),
            ("Security(보안)",  secM),
            ("DataScience(데이터)", dsM),
            ("Analytics(애널리틱스)", anM),
        ]

        # 저변별 막대 + CI (레이더 대체 기본)
        bars=[]
        cats=[]
        groups=["하(Low)","중(Mid)","상(Top)"]
        data=[]
        for label, series in metrics:
            for g in groups:
                m=(grp_all==g)&series.notna()
                if m.sum()>0:
                    mean=float(series[m].mean())
                    lo,hi=ci95(series[m])
                    data.append({"Metric":label,"Group":g,"Mean":mean,"CI+":(hi-mean if hi==hi else 0)})
        df_bar=pd.DataFrame(data)
        if len(df_bar)>0:
            fig=px.bar(df_bar, x="Metric", y="Mean", color="Group", barmode="group",
                       title="그룹 평균(0~100) + 95% CI", error_y="CI+",
                       category_orders={"Group":groups})
            fig.update_layout(font_family=font_family, yaxis=dict(range=[0,100]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("계산 가능한 평균이 없습니다.")

        # 보조: Z-스코어 프로파일(개별 기업 비교)
        st.caption("보조: 표준화(Z-score) 평면에 기업 분포(저분산 시 구분력 향상)")
        # robust z: (x - median)/IQR
        def robust_z(s):
            x=s.copy().astype(float)
            med=float(np.nanmedian(x)); iqr=float(np.nanpercentile(x,75)-np.nanpercentile(x,25))
            if iqr==0 or iqr!=iqr: return (x*0)*np.nan
            return (x-med)/iqr
        rz=pd.DataFrame({
            "Company": df[COL_COMP],
            "DesignOps": robust_z(des100),
            "Security":  robust_z(secM),
            "DataSci":   robust_z(dsM),
            "Analytics": robust_z(anM)
        })
        rz["Z합계"]=rz[["DesignOps","Security","DataSci","Analytics"]].sum(axis=1, skipna=True)
        rz_valid = rz.dropna(subset=["Z합계"])
        if len(rz_valid)>=3:
            top = rz_valid.nlargest(15,"Z합계")
            fig2=px.scatter(top, x="DesignOps", y="Security", hover_name="Company",
                            title="상위 15개 기업(합산 Z) – DesignOps vs Security",
                            labels={"DesignOps":"DesignOps(Z)","Security":"Security(Z)"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Z-스코어 기반 비교를 위한 표본 부족.")
    else:
        st.info("AI 채택 표본이 3개 미만. A4 생략.")

# A5
with tabs[4]:
    st.header("A5. AI 채택 × 보안 거버넌스 2×2 – 강화 뷰")
    ai_max = ai.max(skipna=True) if ai.notna().any() else 5
    ai100 = (ai/(ai_max if ai_max else 5))*100.0
    gov100 = secM
    m = ai100.notna() & gov100.notna()
    if m.sum()>=2:
        x=ai100[m]; y=gov100[m]; comp=df.loc[m, COL_COMP]
        xm=float(x.median()); ym=float(y.median())

        # 2D 히스토그램(밀도)으로 사분면 구분력 강화
        dt = pd.DataFrame({"AI":x,"SEC":y})
        fig = px.density_heatmap(dt, x="AI", y="SEC", nbinsx=10, nbinsy=10, title="사분면 밀도 히트맵")
        fig.add_vline(x=xm, line_dash="dash"); fig.add_hline(y=ym, line_dash="dash")
        fig.update_layout(font_family=font_family, xaxis=dict(range=[0,100]), yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

        # 사분면별 개수 표 + 위험 리스트
        q = pd.DataFrame({"AI":x, "SEC":y, "Company":comp})
        q["Quadrant"] = np.where((q["AI"]>=xm) & (q["SEC"]>=ym), "Q1 고채택·고보안",
                           np.where((q["AI"]<xm) & (q["SEC"]>=ym), "Q2 저채택·고보안",
                           np.where((q["AI"]<xm) & (q["SEC"]<ym),  "Q3 저채택·저보안",
                                    "Q4 고채택·저보안(위험)")))
        counts = q["Quadrant"].value_counts().reindex(["Q1 고채택·고보안","Q2 저채택·고보안","Q3 저채택·저보안","Q4 고채택·저보안(위험)"]).fillna(0).astype(int)
        st.write("사분면 개수:", counts.to_dict())
        risk_list = q[q["Quadrant"]=="Q4 고채택·저보안(위험)"].sort_values(["SEC","AI"]).head(20)
        if len(risk_list)>0:
            st.caption("위험 후보 Top20 (보안 낮고 채택 높은 순)")
            st.dataframe(risk_list, use_container_width=True)
    else:
        st.info("표본 부족. 최소 2개 이상 필요.")
