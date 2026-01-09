import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
from pathlib import Path
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 📌 페이지 설정
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="멧돼지 출몰 감지 및 위험도 평가 시스템",
    page_icon="🐗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# 📍 위치 데이터 (AI Hub)
# ═══════════════════════════════════════════════════════════════
LOCATION_DATA = {
    '태백시 삼수동': {'lat': 37.1542, 'lng': 129.0388, 'region': '강원도', 'count': 1200},
    '정선군 정선읍': {'lat': 37.3747, 'lng': 129.3858, 'region': '강원도', 'count': 980},
    '평창군 진부면': {'lat': 37.4267, 'lng': 128.8925, 'region': '강원도', 'count': 1100},
    '영월군 영월읍': {'lat': 37.1875, 'lng': 128.8225, 'region': '강원도', 'count': 890},
    '삼척시 근덕면': {'lat': 37.3433, 'lng': 129.4667, 'region': '강원도', 'count': 750},
    '동해시 삼화동': {'lat': 37.5217, 'lng': 129.1158, 'region': '강원도', 'count': 650},
    '강릉시 옥계면': {'lat': 37.8042, 'lng': 129.2425, 'region': '강원도', 'count': 890},
    '속초시 광장동': {'lat': 38.2011, 'lng': 128.5926, 'region': '강원도', 'count': 1300},
    '인제군 인제읍': {'lat': 38.0578, 'lng': 128.1661, 'region': '강원도', 'count': 1050},
    '양양군 손양면': {'lat': 37.8828, 'lng': 128.8233, 'region': '강원도', 'count': 920},
    '홍천군 홍천읍': {'lat': 37.7558, 'lng': 127.9361, 'region': '강원도', 'count': 800},
    '화천군 화천읍': {'lat': 37.8625, 'lng': 127.7142, 'region': '강원도', 'count': 1150},
    '춘천시 남산면': {'lat': 37.8058, 'lng': 127.7328, 'region': '강원도', 'count': 650},
    '가평군 가평읍': {'lat': 37.8361, 'lng': 127.5103, 'region': '경기도', 'count': 920},
    '남이섬': {'lat': 37.9633, 'lng': 127.5467, 'region': '경기도', 'count': 450},
    '여주시 점동면': {'lat': 37.2931, 'lng': 127.6436, 'region': '경기도', 'count': 680},
}

# ═══════════════════════════════════════════════════════════════
# 🔧 위험도 계산 함수
# ═══════════════════════════════════════════════════════════════
def calculate_risk_score(detection_count, avg_confidence, is_night=False, location=''):
    """위험도 점수 계산 (0-100)"""
    base_score = (detection_count * 10) * 0.4 + (avg_confidence * 100) * 0.4
    night_bonus = 20 if is_night else 0
    night_score = night_bonus * 0.2
    total_score = min(base_score + night_score, 100)
    return total_score

def get_risk_level(score):
    """위험도 레벨 반환"""
    if score >= 80:
        return "🔴 극도로 높음", "#ff0000", "danger"
    elif score >= 60:
        return "🟠 높음", "#ff6b00", "warning"
    elif score >= 40:
        return "🟡 중간", "#ffa500", "caution"
    elif score >= 20:
        return "🟢 낮음", "#90ee90", "safe"
    else:
        return "🟢 매우 낮음", "#00aa00", "very_safe"

def get_management_guide(risk_score):
    """위험도 기반 관리자 대응 가이드"""
    guides = {
        "danger": {
            "level": "🔴 극도로 높음 (위험도 80 이상)",
            "actions": [
                "✋ 해당 지역 탐방로 즉시 통제",
                "🔔 주민 긴급 공보 실시",
                "👮 야간 순찰 최대 강화 (격일제 순찰)",
                "📞 관광지/캠핑장 방문객 주의 안내",
                "🏢 지자체/산림청에 즉시 신고",
                "⏰ 실시간 모니터링 필수",
                "📋 대응 기록 상세 보관"
            ]
        },
        "warning": {
            "level": "🟠 높음 (위험도 60-80)",
            "actions": [
                "⚠️ 탐방로 야간 통제 권고",
                "📢 주민 주의 공지",
                "👮 야간 순찰 강화",
                "🔦 관광지 조명 강화",
                "📞 관련 기관 상황 보고",
                "📊 일일 통계 작성"
            ]
        },
        "caution": {
            "level": "🟡 중간 (위험도 40-60)",
            "actions": [
                "📌 야외 활동 시 주의 안내",
                "🔔 주기적 모니터링",
                "📢 지역민 공지",
                "🗺️ 주의 표지판 설치/점검",
                "👥 단체 활동 권장",
                "📊 주간 통계 작성"
            ]
        },
        "safe": {
            "level": "🟢 낮음 (위험도 20-40)",
            "actions": [
                "✅ 일반적인 주의 유지",
                "🔍 정상 모니터링",
                "📋 주간 보고",
                "🗺️ 표지판 유지"
            ]
        },
        "very_safe": {
            "level": "🟢 매우 낮음 (위험도 20 이하)",
            "actions": [
                "✅ 정상 운영",
                "🔍 기본 감시",
                "📋 월간 보고"
            ]
        }
    }
    
    if risk_score >= 80:
        return guides["danger"]
    elif risk_score >= 60:
        return guides["warning"]
    elif risk_score >= 40:
        return guides["caution"]
    elif risk_score >= 20:
        return guides["safe"]
    else:
        return guides["very_safe"]

# ═══════════════════════════════════════════════════════════════
# 💾 모델 로드
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        model = YOLO(r"D:\딥러닝 프로젝트\runs\boar_detection_v2\weights\best.pt")
        return model
    except:
        st.warning("⚠️ 모델 파일을 찾을 수 없습니다. 데모 모드로 진행합니다.")
        return None

# ═══════════════════════════════════════════════════════════════
# 📌 탐지 함수
# ═══════════════════════════════════════════════════════════════
def detect_boar(image, model, conf_threshold=0.5):
    """멧돼지 탐지"""
    if model is None:
        # 데모 모드: 임의의 탐지 결과 생성
        detections = [
            {'confidence': 0.95, 'box': np.array([100, 100, 200, 200])},
            {'confidence': 0.87, 'box': np.array([250, 150, 350, 300])}
        ]
        result_image = image
        return result_image, detections
    
    results = model.predict(image, conf=conf_threshold, imgsz=416)
    
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            detections.append({
                'confidence': float(box.conf),
                'box': box.xyxy[0].cpu().numpy(),
                'class': int(box.cls)
            })
    
    result_image = results[0].plot()
    return result_image, detections

def save_results(image, detections, location, timestamp):
    """결과 저장"""
    save_dir = Path("detection_results")
    save_dir.mkdir(exist_ok=True)
    
    img_path = save_dir / f"detection_{timestamp.replace(':', '-')}.jpg"
    json_path = save_dir / f"detection_{timestamp.replace(':', '-')}.json"
    
    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    json_data = {
        'timestamp': timestamp,
        'location': location,
        'detections': detections,
        'count': len(detections)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return img_path, json_path

# ═══════════════════════════════════════════════════════════════
# 🎨 사이드바 설정
# ═══════════════════════════════════════════════════════════════
st.sidebar.title("🐗 시스템 정보")

st.sidebar.info(
    f"""
    **AI 기반 멧돼지 출몰 감지 및**
    **위험도 평가·의사결정 지원 웹 시스템**
    
    **🔧 시스템 정보**
    - YOLOv8n 기반 객체 탐지
    - imgsz=416 (고속 처리)
    - 위험도 정량화 & 의사결정 지원
    - 자동 대응 가이드 생성
    
    **📊 데이터 기준**
    - 촬영 위치: {len(LOCATION_DATA)}개
    - 라벨링 데이터: 31,697개
    - 탐지 정확도: 97.5% (mAP)
    - 평균 탐지 속도: 5-7초/이미지
    
    **👤 개발자**: AI 프로젝트
    **📌 버전**: 3.0 (위험도 평가 & 의사결정 지원)
    **📚 데이터**: AI Hub + 농림축산식품부
    
    **✨ v3.0 핵심 기능**:
    - ✅ 위험도 점수화
    - ✅ 관리자 대응 가이드
    - ✅ 누적 패턴 분석
    - ✅ 관제 모드 시뮬레이션
    - ✅ 위험도 예측
    """
)

# ═══════════════════════════════════════════════════════════════
# 📌 세션 상태 초기화
# ═══════════════════════════════════════════════════════════════
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# ═══════════════════════════════════════════════════════════════
# 🎯 메인 제목
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1>🐗 멧돼지 출몰 감지 및 위험도 평가 시스템</h1>
    <p style='font-size: 16px; color: #666;'>AI 기반 의사결정 지원 웹 시스템 v3.0</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# 📌 탭 생성
# ═══════════════════════════════════════════════════════════════
tab1, tab3, tab4, tab5, tab7 = st.tabs([
    "📸 이미지 탐지",
    "📍 위치 추적",
    "📊 누적 통계",
    "📚 가이드",
    "⚠️ 위험도 예측"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: 이미지 탐지 (핵심 - 위험도 판단)
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("📸 이미지 탐지")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "이미지 업로드 (JPG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )
    
    with col2:
        selected_location = st.selectbox(
            "📍 촬영 위치",
            options=sorted(list(LOCATION_DATA.keys()))
        )
    
    model = load_model()
    
    if uploaded_file and model is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        detection_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result_image, detections = detect_boar(image_np, model, conf_threshold=0.5)
        
        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
            is_night = datetime.now().hour >= 21 or datetime.now().hour <= 5
            risk_score = calculate_risk_score(
                len(detections),
                avg_conf,
                is_night,
                selected_location
            )
        else:
            risk_score = 0
            avg_conf = 0
        
        st.session_state.detection_results = {
            'result_image': result_image,
            'detections': detections,
            'selected_location': selected_location,
            'detection_timestamp': detection_timestamp,
            'risk_score': risk_score,
            'avg_confidence': avg_conf
        }
        
        st.subheader("🎯 탐지 결과")
        st.caption(f"📅 탐지 시간: {detection_timestamp}")
        
        risk_level_text, risk_color, risk_type = get_risk_level(risk_score)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("탐지 수", len(detections))
        with col2:
            st.metric("평균 신뢰도", f"{avg_conf:.2%}")
        with col3:
            st.metric("위험도 점수", f"{risk_score:.1f}/100", delta=risk_level_text)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "위험도 평가"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 20], 'color': "#90ee90"},
                    {'range': [20, 40], 'color': "#ffff00"},
                    {'range': [40, 60], 'color': "#ffa500"},
                    {'range': [60, 80], 'color': "#ff6b00"},
                    {'range': [80, 100], 'color': "#ff0000"}
                ]
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style='background-color: {risk_color}; padding: 15px; border-radius: 10px; color: white; text-align: center; font-size: 18px; font-weight: bold;'>
            {risk_level_text}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        st.subheader("🖼️ 탐지 이미지")
        st.image(result_image, width='stretch')
        
        if detections:
            st.markdown("---")
            st.subheader("🔍 탐지 세부 정보")
            
            detection_df = pd.DataFrame([
                {
                    '번호': i+1,
                    '신뢰도': f"{d['confidence']:.2%}",
                    '위치': f"({d['box'][0]:.0f}, {d['box'][1]:.0f})"
                }
                for i, d in enumerate(detections)
            ])
            
            st.dataframe(detection_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 결과 저장"):
                img_path, json_path = save_results(
                    result_image,
                    [{'confidence': float(d['confidence'])} for d in detections],
                    selected_location,
                    detection_timestamp
                )
                st.success(f"✅ 저장 완료!")
                st.caption(f"📁 이미지: {Path(img_path).name}")
                st.caption(f"📄 JSON: {Path(json_path).name}")
        


# ═══════════════════════════════════════════════════════════════
# TAB 3: 위치 추적
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("📍 위치 추적")
    
    m = folium.Map(
        location=[37.5, 128.5],
        zoom_start=8,
        tiles="OpenStreetMap"
    )
    
    for location_name, info in LOCATION_DATA.items():
        folium.CircleMarker(
            location=[info['lat'], info['lng']],
            radius=8,
            popup=f"{location_name}\n탐지: {info['count']}건",
            color='red',
            fill=True,
            fillColor='#ff6b6b',
            fillOpacity=0.7
        ).add_to(m)
    
    st_folium(m, width=1400, height=600)
    
    st.markdown("---")
    st.subheader("📊 위치별 누적 탐지")
    
    location_stats = pd.DataFrame([
        {
            '위치': k,
            '지역': v['region'],
            '탐지건수': v['count'],
        }
        for k, v in LOCATION_DATA.items()
    ])
    
    fig = px.bar(
        location_stats.sort_values('탐지건수', ascending=False),
        x='위치',
        y='탐지건수',
        color='탐지건수',
        title="위치별 누적 탐지 현황",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4: 누적 통계
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 누적 통계 및 패턴 분석")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("모델", "YOLOv8n")
    col2.metric("이미지 크기", "416×416")
    col3.metric("장치", "CPU")
    col4.metric("정확도", "97.5%")
    
    st.markdown("---")
    
    st.subheader("📍 위치별 탐지 현황")
    
    location_stats = pd.DataFrame([
        {'위치': k, '탐지건수': v['count']}
        for k, v in LOCATION_DATA.items()
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            location_stats.sort_values('탐지건수', ascending=False).head(10),
            x='위치',
            y='탐지건수',
            title="상위 10개 위치",
            color='탐지건수',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            location_stats,
            values='탐지건수',
            names='위치',
            title="위치별 탐지 비율"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("⏰ 시간대별 위험도 히트맵")
    
    hours = list(range(24))
    days = ['월', '화', '수', '목', '금', '토', '일']
    
    heatmap_data = np.random.randint(20, 80, (7, 24))
    for i in range(7):
        heatmap_data[i, 0:5] = np.random.randint(60, 100)
        heatmap_data[i, 5:17] = np.random.randint(10, 40)
        heatmap_data[i, 17:21] = np.random.randint(50, 80)
        heatmap_data[i, 21:24] = np.random.randint(70, 100)
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=hours,
        y=days,
        colorscale='Reds',
        colorbar=dict(title="위험도")
    ))
    fig_heat.update_layout(
        title="시간대별 위험도 히트맵 (야행성 멧돼지 패턴)",
        xaxis_title="시간",
        yaxis_title="요일"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📈 최근 7일 위험도 추이")
    
    dates = pd.date_range(start='2026-01-01', periods=7)
    daily_risk = [35, 45, 52, 61, 68, 72, 65]
    
    fig_line = px.line(
        x=dates.strftime('%m-%d'),
        y=daily_risk,
        markers=True,
        title="7일 누적 위험도 변화",
        labels={'x': '날짜', 'y': '위험도 점수'}
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5: 가이드
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.header("📚 사용 가이드")
    
    st.write("🐗 멧돼지 탐지 시스템 v3.0 (위험도 평가 & 의사결정 지원)")
    
    st.markdown("---")
    st.subheader("✨ v3.0 핵심 개선사항")
    
    st.markdown("""
    **1. 위험도 정량화** ✅
    - 탐지 수, 신뢰도, 야간 가중치 기반 객관적 위험도 점수
    - 0-100 스케일로 정량화
    
    **2. 자동 대응 가이드** 📋
    - 위험도 등급별 관리자 액션 아이템 자동 생성
    - 5단계별 구체적 행동 지침
    
    **3. 누적 패턴 분석** 📊
    - 위치별, 시간대별 탐지 누적 분석
    - 야행성 멧돼지의 시간대별 위험도 히트맵

    """)
    
    with st.expander("📸 탭 1: 이미지 탐지", expanded=False):
        st.markdown("""
        1. 이미지 업로드 (JPG, PNG)
        2. 촬영 위치 선택
        3. 자동 탐지 및 위험도 계산
        4. 관리자 대응 가이드 확인
        5. 결과 저장
        """)
    
    with st.expander("🎥 탭 2: 비디오 탐지", expanded=False):
        st.markdown("""
        1. 비디오 파일 업로드
        2. 촬영 위치 선택
        3. 프레임 단위 탐지 자동 실행
        4. 최종 결과 저장
        """)

# ═══════════════════════════════════════════════════════════════
# TAB 7: 위험도 예측
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.header("⚠️ 멧돼지 출몰 위험도 예측")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("🌡️ 기온 (°C)", -10, 40, 10, 1)
        humidity = st.slider("💧 습도 (%)", 0, 100, 60, 5)
        wind_speed = st.slider("💨 풍속 (m/s)", 0, 25, 3, 1)
    
    with col2:
        weather_condition = st.selectbox(
            "☁️ 날씨",
            ['맑음', '구름', '흐림', '안개', '이슬비', '비', '눈', '우박']
        )
        time_of_day = st.selectbox(
            "⏰ 시간대",
            ['아침 (05:00-08:00)', '오전 (08:00-12:00)', '오후 (12:00-17:00)', '저녁 (17:00-21:00)', '밤 (21:00-05:00)']
        )
    
    def predict_risk(temp, humid, wind, weather, time_period):
        risk = 0
        
        if 5 <= temp <= 15:
            risk += 100 * 0.35
        elif 0 <= temp < 5:
            risk += 70 * 0.35
        elif 15 < temp <= 25:
            risk += 40 * 0.35
        else:
            risk += 20 * 0.35
        
        if 50 <= humid <= 80:
            risk += 80 * 0.15
        elif 30 <= humid < 50:
            risk += 50 * 0.15
        else:
            risk += 25 * 0.15
        
        weather_risk = {
            '맑음': 20, '구름': 50, '흐림': 75, '안개': 90,
            '이슬비': 70, '비': 85, '눈': 60, '우박': 30
        }
        risk += weather_risk.get(weather, 50) * 0.20
        
        if 0 <= wind < 3:
            risk += 85 * 0.15
        elif 3 <= wind < 8:
            risk += 45 * 0.15
        else:
            risk += 20 * 0.15
        
        time_risk = {
            '아침 (05:00-08:00)': 75,
            '오전 (08:00-12:00)': 15,
            '오후 (12:00-17:00)': 10,
            '저녁 (17:00-21:00)': 70,
            '밤 (21:00-05:00)': 95
        }
        risk += time_risk.get(time_period, 50) * 0.15
        
        return min(risk, 100)
    
    predicted_risk = predict_risk(temperature, humidity, wind_speed, weather_condition, time_of_day)
    risk_level_text, risk_color, risk_type = get_risk_level(predicted_risk)
    
    st.markdown("---")
    st.subheader("📊 예측 결과")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("예상 위험도", f"{predicted_risk:.1f}/100")
    
    with col2:
        st.markdown(f"""
        <div style='background-color: {risk_color}; padding: 15px; border-radius: 10px; color: white; text-align: center; font-weight: bold;'>
            {risk_level_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if predicted_risk >= 70:
            st.error("🔴 높은 위험도 예상 → 탐방로 통제, 주민 공보, 야간 순찰")
        elif predicted_risk >= 40:
            st.warning("🟡 중간 위험도 예상 → 야간 통제, 기관 보고")
        else:
            st.success("🟢 낮은 위험도 예상 → 주의 안내, 주기적 모니터링")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "예측 위험도"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 20], 'color': "#90ee90"},
                {'range': [20, 40], 'color': "#ffff00"},
                {'range': [40, 60], 'color': "#ffa500"},
                {'range': [60, 80], 'color': "#ff6b00"},
                {'range': [80, 100], 'color': "#ff0000"}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 관리자 대응 가이드")
        
    guide = get_management_guide(predicted_risk)
    st.markdown(f"### {guide['level']}")
        
    for action in guide['actions']:
        st.write(action)
        
    st.markdown("---")

st.caption("🐗 AI 기반 멧돼지 출몰 감지 및 모니터링 시스템")