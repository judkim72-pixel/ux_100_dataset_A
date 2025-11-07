# UX 100 Dataset • A 모듈 대시보드 (한글/툴팁)

## 실행
```bash
pip install -r requirements.txt
streamlit run A_AutoK.py
```

## 데이터
- 리포지토리 루트에 `ux_100_dataset.xlsx` 배치
- 시트: `Data` (컬럼은 스크립트 상단의 상수명을 참조)

## 폰트 (맑은 고딕)
- Plotly는 **브라우저 폰트**를 사용하므로 대부분 클라이언트에서 정상 렌더됩니다.
- 서버 렌더(이미지 저장 등)까지 한글 폰트 일치가 필요하면 `MalgunGothic.ttf`를 **리포지토리 루트**에 추가하세요.

## 문제 해결
- `ModuleNotFoundError: plotly` → `requirements.txt` 설치 여부 확인
- `FileNotFoundError: ux_100_dataset.xlsx` → 루트 경로 확인
- 컬럼 누락 오류 → 엑셀 `Data` 시트의 컬럼명을 스크립트 상단 상수에 맞춰 수정