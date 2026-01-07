# 🐗 AI 기반 멧돼지 출몰 감지 및 위험도 예측 시스템

딥러닝 기반 객체 탐지 모델(YOLOv8)을 활용하여 멧돼지 출몰 여부를 자동 감지하고,
기상·시간대 정보를 기반으로 출몰 위험도를 예측하여 관리자의 선제적 대응과 의사결정을 지원하는 AI 웹 서비스 프로젝트입니다.

과거에 진행했던 딥러닝 프로젝트를 최신 기술(YOLOv8, Streamlit)과
실무 관점에 맞게 재구성하여 모델 구현을 넘어 서비스 형태로 완성하는 것을 목표로 하였습니다.

---
## 🎬 서비스 시연 영상
[https://github.com/minseo3280-coder/Flask_project/issues/1#issue-3769423576](https://private-user-images.githubusercontent.com/248983211/531126448-4ddea955-a170-46b6-88e4-1d08ede514f1.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcxNjc1MDQsIm5iZiI6MTc2NzE2NzIwNCwicGF0aCI6Ii8yNDg5ODMyMTEvNTMxMTI2NDQ4LTRkZGVhOTU1LWExNzAtNDZiNi04OGU0LTFkMDhlZGU1MTRmMS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjMxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIzMVQwNzQ2NDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lMjdjYzZjY2JjNDg1Yjc5MTM2NjY0NzI4M2IwMTczMjQ1MTA2NTczNzdhOGNhN2VjMzA1ZjIwMmM0YWZhYmY1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.OgUTB0zUJhhWiIL6ZMoYQlFyLahSwYcPaEjY85uXkHw)

---
## 📄 프로젝트 발표 자료

상세 기획, 분석, 개발 과정은 다음 보고서에서 확인하세요:

👉 **[AI 기반 실시간 멧돼지 출몰 감지 및 위험도 예측 시스템.pdf](AI기반 실시간 멧돼지 출몰 감지 및 위험도 예측 시스템.pdf)**

---

## 🛠 기술 스택

- **Language**: Python 3.9+
- **Deep Learning**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Web**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Environment**: Google Colab, VS Code
- **Version Control**: Git, GitHub

---

## 📌 문제 정의

### 현황
- 멧돼지 출몰로 인한 **인명·농작물 피해 지속 증가**
- 기존 대응은 **신고 이후 대응** 중심 → 예방 한계
- 영상 데이터는 존재하지만 **자동 분석 및 판단 시스템 부족**

### 핵심 질문
> **"멧돼지 출몰을 자동으로 감지하고, 위험도를 미리 예측할 수 없을까?"**

---

## 📊 데이터 설명

### 데이터 개요
- **출처**: [AI-Bub - 야생동물 활동 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=645)
- **데이터 형태**: - AI Hub Dataset (31,697 labeled images)
                   - 영상 (MP4) → 프레임 단위 이미지 추출
- **라벨 형식**: YOLO 포맷 (Bounding Box)

### 전처리 전략

```
원본 영상
 └─ 프레임 단위 이미지 추출
    ├─ 이미지 리사이즈 (YOLO 입력 규격)
    ├─ 라벨 정합성 검증
    └─ Train / Validation 분리

```

---

## 🤖 모델 설계 및 학습

### 모델 선택: YOLOv8n

| 항목 | 선택 이유 |
|------|----------|---------|-----------|--------|----------|------|
| **경량 모델** | 웹 서비스 환경에 적합|
| **빠른 추론 속도** | 실시간 확장 가능성|
| **성능 대비 효율** | 정확도 손실 최소|


- **yolov8_model_sizes 그래프 추가**


> YOLOv8m 대비 정확도 차이는 크지 않으나,
**추론 속도·자원 효율·웹 안정성 측면에서 YOLOv8n이 더 적합**하다고 판단하였습니다.
>

### 학습 설정
- **Epoch**: 50
- **Optimizer**: YOLOv8 기본 설정
- **Overfitting 방지**: Epoch별 성능 모니터링
---

**최적 파라미터**: `n_estimators=150`, `max_depth=20`, `min_samples_split=10`, `min_samples_leaf=2`

---

## 📈 모델 분석 결과
### Epoch별 성능 분석

- 50 Epoch 이후 성능 수렴 확인
- 추가 학습 시 정확도 개선 폭 미미
- 과적합 위험 대비 **50 Epoch** 최적 선택

- **epch 그래프 그림 추가**


### 모델 비교 요약
- YOLOv8n: 빠른 추론 + 안정적 성능
- 웹 환경 적용에 적합한 실무형 모델


---
## 📂 프로젝트 구조

```
boar-detection-system/
│
├── app.py                          # 메인 Streamlit 애플리케이션
├── boar_detection_colab.py         # Google Colab 학습 스크립트
├── extract_locations.py            # 위치 데이터 추출 스크립트
│
├── data/
│   ├── all_locations.json          # 16개 위치 메타데이터
│   ├── locations_summary.csv       # 위치별 통계
│   └── results.csv                 # 모델 성능 결과
│
├── weights/
│   └── best.pt                     # YOLOv8m 학습 완료 모델
│
├── detection_results/              # 탐지 결과 저장 폴더
│   ├── *.jpg                       # 탐지된 이미지
│   └── *.json                      # 탐지 메타데이터
│
├── requirements.txt                # 의존성 라이브러리
├── README.md                       # 프로젝트 문서 (본 파일)
└── LICENSE                         # MIT 라이선스
```

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/minseo3280-coder/boar-detection-system.git
cd boar-detection-system

# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ 모델 가중치 다운로드

```bash
# 최적화된 YOLO 모델 다운로드
# (프로젝트 루트에 weights/ 폴더 생성 후 best.pt 저장)
```

### 3️⃣ 애플리케이션 실행

```bash
# Streamlit 앱 실행
streamlit run app.py

# 웹 브라우저 자동 열림
# http://localhost:8501
```

---

## 📊 주요 기능 상세 설명

### TAB 1️⃣: 📸 이미지 탐지 (핵심 기능)

**기능:**
- JPG/PNG 이미지 업로드
- 자동 멧돼지 탐지 및 바운딩박스 표시
- 위험도 점수 즉시 계산
- 관리자 대응 가이드 자동 생성
- 탐지 결과 저장 (이미지 + JSON)

**위험도 평가:**
```
🔴 극도로 높음 (80 이상)  → 탐방로 즉시 통제, 주민 긴급 공보
🟠 높음 (60-80)           → 야간 통제, 기관 보고
🟡 중간 (40-60)           → 주의 안내, 주기적 모니터링
🟢 낮음 (20-40)           → 정상 모니터링 유지
🟢 매우 낮음 (20 이하)    → 정상 운영
```

### TAB 2️⃣: 📍 위치 추적

**기능:**
- 인터랙티브 Folium 지도
- 16개 촬영 위치 마커 표시
- 위치별 누적 탐지 건수 팝업
- 막대 그래프로 위치별 통계

### TAB 3️⃣: 📊 누적 통계

**분석 항목:**
1. **위치별 탐지 현황**
   - 상위 10개 위치 (막대 그래프)
   - 전체 위치별 비율 (원형 그래프)

2. **시간대별 위험도 히트맵**
   - 요일 × 시간대 (7×24 격자)
   - 야행성 멧돼지 패턴 분석

3. **7일 위험도 추이**
   - 일자별 누적 위험도 라인 차트
   - 추세 분석

### TAB 4️⃣: 🎥 실시간 대시보드 (산림청 관제 센터)

**메인 기능:**
- 🟢 **시스템 상태** (5개 메트릭)
  - 전체 모니터링 위치 수
  - 당일 누적 탐지 건수
  - 현재 위험도 평가
  - 고위험 지역 수
  - 최근 업데이트 시간

- 📢 **실시간 알림 로그**
  - 최근 10건 탐지 기록
  - 시간, 위치, 탐지 수, 신뢰도, 위험도 표시

- 📍 **위치별 위험도 현황**
  - 고위험 지역 (60 이상)
  - 주의 필요 지역 (40-60)

- ⚙️ **모니터링 설정**
  - 새로고침 간격 조정
  - 경보 임계값 설정

- 📋 **일일 보고서**
  - 자동 생성 & 이메일 발송 (매일 18:00)

### TAB 5️⃣: 📚 가이드

- 시스템 사용 방법 상세 설명
- 각 탭별 기능 설명
- 사용자 시나리오

### TAB 6️⃣: ⚠️ 위험도 예측

**입력 파라미터:**
- 🌡️ 기온 (-10~40°C)
- 💧 습도 (0~100%)
- 💨 풍속 (0~25m/s)
- ☁️ 날씨 (맑음/구름/흐림/안개/비/눈 등)
- ⏰ 시간대 (아침/오전/오후/저녁/밤)

**출력:**
- 예측 위험도 (0-100)
- 위험 등급 표시
- 요인별 기여도 분석
- **자동 행동 가이드 + 체크리스트** ⭐

---

## 📈 모델 성능

### 학습 결과

| 지표 | 값 |
|------|-----|
| **mAP (Mean Average Precision)** | 97.5% |
| **Precision** | 96.8% |
| **Recall** | 98.2% |
| **탐지 속도** | 5-7초/이미지 |
| **학습 데이터** | 31,697개 이미지 |
| **촬영 위치** | 16개 (강원도 일대) |

### 데이터셋

```
데이터 출처: AI Hub (야생동물 활동 영상 데이터)
학습:검증:테스트 = 70:15:15

위치 분포:
- 태백시, 정선군, 평창군, 영월군 (강원도)
- 삼척시, 동해시, 강릉시, 속초시
- 인제군, 양양군, 홍천군, 화천군, 춘천시
- 가평군, 남이섬, 여주시 (경기도)
```

---

## 💡 사용 사례

### 시나리오 1: 실시간 탐지

```
1. 캠프장 CCTV 영상 촬영
2. 이미지 탐지 탭에서 업로드
3. 즉시 위험도 평가 (예: 68점 → 🟠 높음)
4. 자동 대응 가이드 제시:
   - ⚠️ 탐방로 야간 통제 권고
   - 📢 주민 주의 공지
   - 👮 야간 순찰 강화
   - 🔦 관광지 조명 강화
5. 관리자가 체크리스트 기반으로 즉시 조치
```

### 시나리오 2: 사전 예측

```
1. 내일 날씨 입력 (기온: 8°C, 습도: 70%, 흐림, 밤)
2. 예측 위험도: 82.5점 → 🔴 극도로 높음
3. 자동 행동 가이드:
   - ✋ 탐방로 즉시 통제
   - 🔔 주민 긴급 공보
   - 체크리스트로 사전 준비 완료
4. 관리자가 미리 대응 계획 수립
```

---

## 🔧 설치 & 의존성

### requirements.txt
```
streamlit==1.0+
opencv-python==4.5+
numpy==1.20+
pandas==1.3+
pillow==8.0+
folium==0.12+
streamlit-folium==0.5+
plotly==5.0+
ultralytics==8.0+
torch==1.10+
torchvision==0.11+
```

### 설치 명령어
```bash
pip install -r requirements.txt
```

---

## 📱 웹 인터페이스

### 주요 UI 특징

- **반응형 레이아웃** (Wide 모드)
- **직관적 탭 네비게이션** (6개 탭)
- **실시간 인터랙티브 차트** (Plotly)
- **지도 기반 위치 시각화** (Folium)
- **즉시 피드백** (위험도 게이지, 색상 코드)

### 색상 체계

```
🟢 Green (#90ee90)    → 낮은 위험도 (20 이하)
🟡 Yellow (#ffff00)   → 중간 위험도 (20-40)
🟠 Orange (#ff6b00)   → 높은 위험도 (60-80)
🔴 Red (#ff0000)      → 극도로 높음 (80 이상)
```

---

## 🔐 데이터 보안 & 저장

### 저장 형식
```
detection_results/
├── detection_2026-01-07-14-45-22.jpg  (탐지 이미지)
└── detection_2026-01-07-14-45-22.json (메타데이터)

JSON 예시:
{
  "timestamp": "2026-01-07 14:45:22",
  "location": "태백시 삼수동",
  "detections": [
    {"confidence": 0.95},
    {"confidence": 0.87}
  ],
  "count": 2
}
```

### 지역 저장소 (로컬)
- 이미지: JPG 포맷 (OpenCV)
- 메타데이터: JSON 포맷
- 자동 폴더 생성

---

## 📊 데이터 분석

### 위험도 계산 공식

```python
위험도 = (탐지 수 × 10) × 0.4 + (신뢰도 × 100) × 0.4 + 야간 가중치 × 0.2

야간 가중치:
  - 21:00~05:00: +20점
  - 기타: 0점

최종 위험도 = min(계산값, 100)
```

### 예측 모델 (다중 요인)

```
예측 위험도 = 
  기온 기여도(35%) 
  + 습도 기여도(15%)
  + 날씨 기여도(20%)
  + 풍속 기여도(15%)
  + 시간대 기여도(15%)
```

---

## 🎓 학습 및 개발 과정

### 단계별 진행

1. **데이터 수집** (AI Hub)
   - 31,697개 야생동물 활동 영상 이미지
   - 16개 지역 촬영
   - 전처리 및 라벨링

2. **모델 학습** (Google Colab)
   - YOLOv8m 파인튜닝
   - 70:15:15 데이터 분할
   - 50 에포크 학습
   - 97.5% 정확도 달성

3. **웹 애플리케이션 개발**
   - Streamlit 프레임워크
   - 6개 탭 기능 구현
   - 대시보드 UI 설계

4. **배포 준비**
   - 모델 경량화 (best.pt)
   - 요구 라이브러리 정리
   - 문서화 완성

---

## 🚀 배포 & 확장

### Streamlit Cloud 배포

```bash
# 1. GitHub에 푸시
git push origin main

# 2. Streamlit Cloud 대시보드에서 배포
# https://share.streamlit.io/

# 3. 공개 URL 생성
# https://boar-detection.streamlit.app/
```

### 클라우드 배포 옵션

| 플랫폼 | 장점 | 비용 |
|--------|------|------|
| **Streamlit Cloud** | 무료, 쉬운 배포 | 무료 |
| **AWS** | 높은 확장성 | 유료 |
| **Google Cloud** | 대규모 처리 | 유료 |
| **Heroku** | 간단한 배포 | 유료 |


### 향후 개선 방향

- [ ] 실시간 RTSP 스트림 처리
- [ ] 관리자 알림 시스템 추가
- [ ] 기상 API 연동으로 위험도 예측 고도화
- [ ] 다양한 야생동물(고라니 등) 탐지 확장
- [ ] 예측 모델 고도화

---


## ⭐ 특별 감사의 말

- AI Hub (데이터 제공)
- Ultralytics (YOLOv8)
- Streamlit (웹 프레임워크)

---

**최근 업데이트:** 2026-01-07  
**상태:** ✅ Active Development

---

## 🎉 마지막으로

이 프로젝트가 산림청 및 지자체의 **멧돼지 출몰 예방 및 관리**에 도움이 되기를 바랍니다!

⭐ **혹시 도움이 되었다면 Star를 눌러주세요!** ⭐




















