import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 필수 라이브러리
try:
    from tqdm import tqdm
except ImportError:
    print("📦 tqdm 설치 중...")
    os.system('pip install -q tqdm')
    from tqdm import tqdm

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("📦 scikit-learn 설치 중...")
    os.system('pip install -q scikit-learn')
    from sklearn.model_selection import train_test_split

try:
    from ultralytics import YOLO
except ImportError:
    print("📦 ultralytics 설치 중...")
    os.system('pip install -q ultralytics')
    from ultralytics import YOLO

from PIL import Image


# ============================================================================
# STEP 1: 환경 설정 및 경로 검증 (✨ 개선됨)
# ============================================================================

def setup_environment():
    """환경 설정 및 경로 검증"""
    print("\n" + "="*70)
    print("📦 Step 1: 환경 설정 중...")
    print("="*70)
    
    # ✨ CRITICAL: 한글 경로를 정확하게 처리
    # D:\딥러닝 프로젝트 로 설정 (공백 없음!)
    base_dir = Path(r'D:\딥러닝 프로젝트')
    
    print(f"\n✅ Base Directory 설정:")
    print(f"   {base_dir}")
    print(f"   Type: {type(base_dir)}")
    
    # 경로 딕셔너리
    dirs = {
        'base': base_dir,
        'data': base_dir / 'data',
        'train_label': base_dir / 'data' / '1.Training' / '라벨링데이터',
        'train_image': base_dir / 'data' / '1.Training' / '원천데이터',
        'val_label': base_dir / 'data' / '2.Validation' / '라벨링데이터',
        'val_image': base_dir / 'data' / '2.Validation' / '원천데이터',
    }
    
    # 경로 검증
    print(f"\n🔍 데이터 경로 검증:")
    all_exist = True
    for key, path in dirs.items():
        if key == 'base':
            continue
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {key:15} : {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print(f"\n❌ 오류: 일부 폴더가 없습니다!")
        print(f"\n✅ 이 명령으로 폴더 생성하세요:")
        print(f"""
mkdir "D:\\딥러닝 프로젝트\\data\\1.Training\\원천데이터"
mkdir "D:\\딥러닝 프로젝트\\data\\1.Training\\라벨링데이터"
mkdir "D:\\딥러닝 프로젝트\\data\\2.Validation\\원천데이터"
mkdir "D:\\딥러닝 프로젝트\\data\\2.Validation\\라벨링데이터"
        """)
        return None, None
    
    print(f"\n✅ 모든 경로 검증 완료!")
    
    return base_dir, dirs


# ============================================================================
# STEP 2: JSON → YOLO 형식 변환 함수
# ============================================================================

def json_to_yolo(json_path, image_path):
    """JSON 형식을 YOLO txt 형식으로 변환"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지 정보 추출
        if not data.get('images') or len(data['images']) == 0:
            return None
        
        img_info = data['images'][0]
        img_width = img_info.get('width', 1920)
        img_height = img_info.get('height', 1080)
        
        # 바운딩 박스 추출
        yolo_lines = []
        annotations = data.get('annotations', [])
        
        for annotation in annotations:
            bbox = annotation.get('bbox', [])
            
            if not bbox or len(bbox) < 2:
                continue
            
            try:
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                if 0 < center_x < 1 and 0 < center_y < 1 and width > 0.001 and height > 0.001:
                    yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            except (IndexError, TypeError, ValueError):
                continue
        
        return yolo_lines if yolo_lines else None
        
    except Exception as e:
        print(f"   ⚠️ JSON 변환 실패 ({json_path.name}): {str(e)}")
        return None


# ============================================================================
# STEP 3: 데이터 처리 (✨ 경로 처리 개선)
# ============================================================================

def process_data(dirs):
    """데이터 처리 및 YOLO 형식으로 변환"""
    
    print("\n" + "="*70)
    print("📊 Step 2: 데이터 처리 중...")
    print("="*70)
    
    # ✨ CRITICAL: Path 객체로 모든 경로 설정
    yolo_root = dirs['base'] / 'yolo_dataset'
    
    print(f"\n📁 YOLO 데이터셋 경로:")
    print(f"   {yolo_root}")
    print(f"   절대 경로: {yolo_root.resolve()}")
    
    # 1. 디렉토리 생성
    print(f"\n📁 YOLO 디렉토리 구조 생성...")
    splits = ['train', 'val', 'test']
    for split in splits:
        (yolo_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_root / split / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {split}/ 디렉토리 생성")
    
    # 2. 데이터 수집
    print(f"\n📋 데이터 수집 중...")
    all_data = []
    
    # Training 데이터
    print(f"\n   📁 Training 데이터:")
    train_label_dir = dirs['train_label']
    train_image_dir = dirs['train_image']
    
    if train_label_dir.exists():
        train_json_files = list(train_label_dir.glob('*.json'))
        print(f"      Found: {len(train_json_files)}개 JSON 파일")
        
        if len(train_json_files) == 0:
            print(f"      ❌ JSON 파일을 찾을 수 없습니다!")
            print(f"      경로 확인: {train_label_dir}")
        
        for json_file in train_json_files:
            image_name = json_file.stem
            jpg_file = train_image_dir / f"{image_name}.jpg"
            
            if not jpg_file.exists():
                for ext in ['.JPG', '.png', '.PNG']:
                    alt_jpg = train_image_dir / f"{image_name}{ext}"
                    if alt_jpg.exists():
                        jpg_file = alt_jpg
                        break
                else:
                    continue
            
            yolo_content = json_to_yolo(json_file, jpg_file)
            if yolo_content:
                all_data.append({
                    'image': str(jpg_file),
                    'label': str(json_file),
                    'split': 'train'
                })
        
        train_count = len([d for d in all_data if d['split'] == 'train'])
        print(f"      ✅ 처리됨: {train_count}개")
    else:
        print(f"      ❌ 폴더 없음: {train_label_dir}")
    
    # Validation 데이터
    print(f"\n   📁 Validation 데이터:")
    val_label_dir = dirs['val_label']
    val_image_dir = dirs['val_image']
    
    if val_label_dir.exists():
        val_json_files = list(val_label_dir.glob('*.json'))
        print(f"      Found: {len(val_json_files)}개 JSON 파일")
        
        if len(val_json_files) == 0:
            print(f"      ❌ JSON 파일을 찾을 수 없습니다!")
            print(f"      경로 확인: {val_label_dir}")
        
        for json_file in val_json_files:
            image_name = json_file.stem
            jpg_file = val_image_dir / f"{image_name}.jpg"
            
            if not jpg_file.exists():
                for ext in ['.JPG', '.png', '.PNG']:
                    alt_jpg = val_image_dir / f"{image_name}{ext}"
                    if alt_jpg.exists():
                        jpg_file = alt_jpg
                        break
                else:
                    continue
            
            yolo_content = json_to_yolo(json_file, jpg_file)
            if yolo_content:
                all_data.append({
                    'image': str(jpg_file),
                    'label': str(json_file),
                    'split': 'val'
                })
        
        val_count = len([d for d in all_data if d['split'] == 'val'])
        print(f"      ✅ 처리됨: {val_count}개")
    else:
        print(f"      ❌ 폴더 없음: {val_label_dir}")
    
    # 3. 데이터 분할
    if len(all_data) == 0:
        print(f"\n❌ 오류: 처리된 데이터가 없습니다!")
        print(f"   확인 사항:")
        print(f"   1️⃣ JSON 파일 확인: {train_label_dir}")
        print(f"   2️⃣ JPG 파일 확인: {train_image_dir}")
        print(f"   3️⃣ 파일명 일치 확인")
        return None
    
    print(f"\n📊 데이터 분할:")
    train_data = [d for d in all_data if d['split'] == 'train']
    val_data = [d for d in all_data if d['split'] == 'val']
    
    if len(val_data) >= 10:
        val_data, test_data = train_test_split(
            val_data, 
            test_size=0.3, 
            random_state=42
        )
    else:
        test_data = []
    
    print(f"   Train: {len(train_data)}개 (70%)")
    print(f"   Val:   {len(val_data)}개 (20%)")
    print(f"   Test:  {len(test_data)}개 (10%)")
    
    # 4. 파일 복사 및 변환
    def copy_and_convert(data_list, split_name):
        print(f"\n🔄 {split_name.upper()} 데이터 복사 및 변환...")
        
        success_count = 0
        for item in tqdm(data_list, desc=f"Processing {split_name}"):
            try:
                json_path = Path(item['label'])
                image_path = Path(item['image'])
                stem = image_path.stem
                
                # 이미지 복사
                dest_image = yolo_root / split_name / 'images' / f"{stem}.jpg"
                shutil.copy(str(image_path), str(dest_image))
                
                # 라벨 변환 및 저장
                yolo_lines = json_to_yolo(json_path, image_path)
                if yolo_lines:
                    dest_label = yolo_root / split_name / 'labels' / f"{stem}.txt"
                    with open(dest_label, 'w') as f:
                        f.writelines(yolo_lines)
                    success_count += 1
            
            except Exception as e:
                continue
        
        print(f"   ✅ {split_name.upper()} 완료: {success_count}개")
        return success_count
    
    copy_and_convert(train_data, 'train')
    copy_and_convert(val_data, 'val')
    if test_data:
        copy_and_convert(test_data, 'test')
    
    print(f"\n✅ 데이터 처리 완료!")
    
    return yolo_root


# ============================================================================
# STEP 4: YOLO data.yaml 생성 (✨ 경로 정규화)
# ============================================================================

def create_yaml(yolo_root):
    """YOLO 설정 파일 생성"""
    
    print("\n" + "="*70)
    print("⚙️  Step 3: YOLO data.yaml 생성...")
    print("="*70)
    
    # ✨ CRITICAL: 경로를 정규화하고 forward slash 사용
    yolo_path = yolo_root.resolve()
    train_path = (yolo_path / 'train' / 'images').as_posix()
    val_path = (yolo_path / 'val' / 'images').as_posix()
    test_path = (yolo_path / 'test' / 'images').as_posix()
    
    # data.yaml 내용
    yaml_content = f"""path: {yolo_path.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['boar']
"""
    
    yaml_path = yolo_root / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✅ data.yaml 생성 완료")
    print(f"   위치: {yaml_path}")
    print(f"   절대 경로: {yaml_path.resolve()}")
    print(f"   내용:\n{yaml_content}")
    
    return yaml_path


# ============================================================================
# STEP 5: 모델 학습 (✨ 경로를 문자열로 변환)
# ============================================================================

def train_model(yaml_path, yolo_root):
    """YOLOv8 모델 학습 (CPU 사용!)"""
    
    print("\n" + "="*70)
    print("🎓 Step 4: 모델 학습 시작...")
    print("="*70)
    
    print(f"\n📦 YOLOv8n 모델 로드 중...")
    try:
        model = YOLO('yolov8n.pt')
        print(f"   ✅ 모델 로드 완료")
    except Exception as e:
        print(f"   ❌ 모델 로드 실패: {str(e)}")
        return None
    
    print(f"\n📚 학습 설정:")
    print(f"   데이터셋: {yaml_path}")
    print(f"   에포크: 10 (테스트용)")
    print(f"   배치 크기: 8")
    print(f"   이미지 크기: 416x416")
    print(f"   학습 장치: CPU")
    print(f"\n   ⏱️  예상 시간: 30분 ~ 1시간")
    
    try:
        # ✨ CRITICAL: yaml_path를 문자열로 변환!
        yaml_str = str(yaml_path.resolve())
        project_str = str((yolo_root.parent / 'runs').resolve())
        
        print(f"\n   데이터셋 경로 (문자열): {yaml_str}")
        print(f"   프로젝트 경로 (문자열): {project_str}")
        
        results = model.train(
            data=yaml_str,              # ✨ 문자열로 변환!
            epochs=50,
            imgsz=416,
            batch=8,
            patience=20,
            device='cpu',
            save=True,
            project=project_str,        # ✨ 문자열로 변환!
            name='boar_detection_v2_epoch50',
            plots=True,
            verbose=False,
            workers=0,
        )
        
        print(f"\n✅ 학습 완료!")
        return results
    
    except Exception as e:
        print(f"\n❌ 학습 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STEP 6: 모델 평가
# ============================================================================

def evaluate_model(yolo_root):
    """모델 평가"""
    
    print("\n" + "="*70)
    print("📊 Step 5: 모델 평가...")
    print("="*70)
    
    best_model_path = yolo_root.parent / 'runs' / 'boar_detection_v2_epoch50' / 'weights' / 'best.pt'
    
    print(f"   찾는 경로: {best_model_path}")
    print(f"   절대 경로: {best_model_path.resolve()}")
    
    if not best_model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다")
        print(f"   존재하는지 확인: {best_model_path.exists()}")
        return None
    
    print(f"✅ 모델 찾음!")
    
    try:
        best_model = YOLO(str(best_model_path))
        print(f"\n🔄 평가 중 (CPU)...")
        
        results = best_model.val(device='cpu', imgsz=416)
        
        print(f"\n📊 평가 결과:")
        if hasattr(results, 'box'):
            print(f"   mAP@0.5:     {results.box.map50:.4f}")
            print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
            print(f"   Precision:   {results.box.mp:.4f}")
            print(f"   Recall:      {results.box.mr:.4f}")
        else:
            print(f"   평가 결과: {results}")
        
        return best_model_path
    
    except Exception as e:
        print(f"⚠️  평가 중 오류: {str(e)}")
        return best_model_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """메인 함수"""
    
    print("\n" + "="*70)
    print("🐗 멧돼지 탐지 시스템")
    print("="*70)
    print(f"   환경: VS Code 로컬 (D 드라이브)")
    print(f"   장치: CPU")
    print(f"   시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   수정사항: 경로 오류 완전 해결!")
    print("="*70)
    
    try:
        # ✨ Step 1: 환경 설정
        print("\n🔍 Step 1: 환경 설정...")
        base_dir, dirs = setup_environment()
        if base_dir is None:
            return
        
        # ✨ Step 2: 데이터 처리
        print("\n📊 Step 2: 데이터 처리...")
        yolo_root = process_data(dirs)
        if yolo_root is None:
            return
        
        # ✨ Step 3: YAML 생성
        print("\n⚙️  Step 3: YAML 생성...")
        yaml_path = create_yaml(yolo_root)
        
        # ✨ Step 4: 모델 학습
        print("\n🎓 Step 4: 모델 학습...")
        results = train_model(yaml_path, yolo_root)
        
        # ✨ Step 5: 모델 평가
        print("\n📊 Step 5: 모델 평가...")
        best_model_path = evaluate_model(yolo_root)
        
        # ✨ 완료 정보
        print("\n" + "="*70)
        print("🎉 완료!")
        print("="*70)
        
        if best_model_path:
            print(f"\n✅ 모델 저장 위치:")
            print(f"   {best_model_path}")
        
        print(f"\n📂 생성된 파일:")
        print(f"   YOLO 데이터셋: {yolo_root}")
        print(f"   학습 결과: {yolo_root.parent / 'runs' / 'boar_detection_v2_epoch50'}")
        
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
    
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()