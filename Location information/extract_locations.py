# 📍 AI Hub 라벨링 데이터에서 장소 정보 자동 추출 스크립트
# 3만 개의 JSON 파일을 한 번에 처리합니다!

import json
import os
import csv
from pathlib import Path
from collections import defaultdict

# 설정
JSON_FOLDER = "D:\\딥러닝 프로젝트\\data\\2.Validation\\라벨링데이터"  # 당신의 JSON 파일 폴더 경로로 변경
OUTPUT_CSV = "locations_summary1.csv"
OUTPUT_JSON = "all_locations1.json"

# 데이터 저장할 딕셔너리
locations_data = defaultdict(list)
all_detections = []

print("🔍 JSON 파일 스캔 중...")
json_files = list(Path(JSON_FOLDER).glob("**/*.json"))
total_files = len(json_files)
print(f"총 {total_files}개 파일 발견\n")

# 진행률 표시
for idx, json_file in enumerate(json_files, 1):
    if idx % 1000 == 0:
        print(f"진행 중... {idx}/{total_files} ({idx/total_files*100:.1f}%)")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # images 정보에서 위치 추출
        if 'images' in data:
            for img in data['images']:
                location = img.get('location', 'Unknown')
                gps = img.get('GPS', 'No GPS')
                file_name = img.get('file_name', '')
                date_created = img.get('date_created', '')
                
                # 위치별로 데이터 정리
                locations_data[location].append({
                    'file_name': file_name,
                    'GPS': gps,
                    'date_created': date_created,
                    'json_file': json_file.name
                })
                
                # 전체 탐지 정보 저장 (annotation 포함)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        all_detections.append({
                            'location': location,
                            'GPS': gps,
                            'file_name': file_name,
                            'species': ann.get('species', ''),
                            'category_name': ann.get('category_name', ''),
                            'date_created': date_created
                        })
    
    except json.JSONDecodeError:
        print(f"⚠️ JSON 파싱 오류: {json_file.name}")
    except Exception as e:
        print(f"❌ 오류: {json_file.name} - {str(e)}")

print(f"\n✅ 처리 완료! 총 {total_files}개 파일 분석됨")

# 1️⃣ 장소별 요약 CSV 저장
print("\n📊 1단계: 장소별 요약 CSV 생성 중...")
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['장소명', 'GPS좌표', '데이터개수', '대표파일명'])
    
    for location in sorted(locations_data.keys()):
        items = locations_data[location]
        gps = items[0]['GPS'] if items else 'No GPS'
        count = len(items)
        first_file = items[0]['file_name'] if items else ''
        
        writer.writerow([location, gps, count, first_file])

print(f"✅ {OUTPUT_CSV} 저장 완료 ({len(locations_data)}개 장소)")

# 2️⃣ 전체 탐지 정보 JSON 저장
print(f"\n📊 2단계: 전체 탐지 정보 JSON 생성 중...")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump({
        'total_detections': len(all_detections),
        'unique_locations': len(locations_data),
        'locations': list(locations_data.keys()),
        'detections': all_detections[:1000]  # 처음 1000개만 미리보기
    }, f, ensure_ascii=False, indent=2)

print(f"✅ {OUTPUT_JSON} 저장 완료")

# 3️⃣ 통계 출력
print("\n" + "="*50)
print("📈 통계 요약")
print("="*50)
print(f"✓ 총 라벨링 파일: {total_files}개")
print(f"✓ 탐지된 멧돼지: {len(all_detections)}개")
print(f"✓ 촬영 장소: {len(locations_data)}개")
print(f"\n🗺️ 장소별 데이터 개수 (상위 10):")

sorted_locations = sorted(
    locations_data.items(), 
    key=lambda x: len(x[1]), 
    reverse=True
)[:10]

for location, items in sorted_locations:
    gps = items[0]['GPS'] if items else 'No GPS'
    print(f"  • {location:20} | {len(items):5}개 | GPS: {gps}")

print("\n💾 생성된 파일:")
print(f"  1. {OUTPUT_CSV} - 장소별 요약 (엑셀로 열 수 있음)")
print(f"  2. {OUTPUT_JSON} - 전체 탐지 정보 (JSON 형식)")
