import os
import numpy as np

base_dir = '/mnt/samsung_ssd/hyunchang/inference_results'
files = [
    'Result_radar2_v14.txt',
    'Result_radar1_v4.txt',
    'Result_lidar_v4.txt'
]

# 사용자가 말한 3열은 프로그래밍 인덱스로 2입니다 (0, 1, 2).
# 만약 에러가 또 나거나 값이 이상하면 이 값을 3으로 바꿔보세요.
COLUMN_INDEX = 2 

def analyze_file(file_path):
    print(f"[*] Analyzing: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"    Error: File not found at {file_path}")
        return

    try:
        # skiprows=1 : 첫 번째 줄(헤더)을 건너뜀
        # usecols=COLUMN_INDEX : 지정한 열만 읽음
        data = np.loadtxt(file_path, skiprows=1, usecols=COLUMN_INDEX)
        
        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        std_val = np.std(data)

        print(f"    -> Mean: {mean_val:.6f}")
        print(f"    -> Min : {min_val:.6f}")
        print(f"    -> Max : {max_val:.6f}")
        print(f"    -> Variation (Max-Min): {max_val - min_val:.6f}")
        
    except IndexError:
        print(f"    Error: {COLUMN_INDEX}번째 열이 존재하지 않습니다. 인덱스를 확인해주세요.")
    except ValueError as e:
        print(f"    Error parsing file: {e}")
        print("    (Tip: 데이터 중간에 문자가 섞여 있거나, 열 번호가 틀렸을 수 있습니다.)")
    except Exception as e:
        print(f"    Unexpected Error: {e}")
    print("-" * 50)

if __name__ == "__main__":
    print(f"Target Directory: {base_dir}")
    print(f"Reading Column Index: {COLUMN_INDEX} (3rd column)\n")
    print("-" * 50)
    
    for file_name in files:
        full_path = os.path.join(base_dir, file_name)
        analyze_file(full_path)