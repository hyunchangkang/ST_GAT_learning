import os
import numpy as np

base_dir = '/mnt/samsung_ssd/hyunchang/inference_results'
files = [
    'Result_radar2_v11.txt',
    'Result_radar1_v11.txt',
    'Result_lidar_v11.txt'
]

# 사용자가 말한 3열은 프로그래밍 인덱스로 2입니다 (0, 1, 2).
# 만약 에러가 또 나거나 값이 이상하면 이 값을 3으로 바꿔보세요.
COLUMN_INDEX = 2 
SIGMA_NORM_INDEX = 3

def analyze_file(file_path):
    print(f"[*] Analyzing: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"    Error: File not found at {file_path}")
        return

    try:
        # skiprows=1 : 첫 번째 줄(헤더)을 건너뜀
        # usecols=[COLUMN_INDEX, SIGMA_NORM_INDEX] : sigma, sigma_norm 읽음
        data = np.loadtxt(file_path, skiprows=1, usecols=[COLUMN_INDEX, SIGMA_NORM_INDEX])
        sigma = data[:, 0]
        sigma_norm = data[:, 1]
        
        mean_val = np.mean(sigma)
        min_val = np.min(sigma)
        max_val = np.max(sigma)
        std_val = np.std(sigma)

        print(f"    [Sigma] Mean: {mean_val:.6f}")
        print(f"    [Sigma] Min : {min_val:.6f}")
        print(f"    [Sigma] Max : {max_val:.6f}")
        print(f"    [Sigma] Variation (Max-Min): {max_val - min_val:.6f}")

        mean_norm = np.mean(sigma_norm)
        min_norm = np.min(sigma_norm)
        max_norm = np.max(sigma_norm)
        std_norm = np.std(sigma_norm)
        frac_hi = np.mean(sigma_norm >= 0.999)
        frac_lo = np.mean(sigma_norm <= 0.001)

        print(f"    [SigmaNorm] Mean: {mean_norm:.6f}")
        print(f"    [SigmaNorm] Min : {min_norm:.6f}")
        print(f"    [SigmaNorm] Max : {max_norm:.6f}")
        print(f"    [SigmaNorm] Std : {std_norm:.6f}")
        print(f"    [SigmaNorm] Frac>=0.999: {frac_hi:.3f}")
        print(f"    [SigmaNorm] Frac<=0.001: {frac_lo:.3f}")
        
    except IndexError:
        print(f"    Error: 필요한 열이 존재하지 않습니다. 인덱스를 확인해주세요.")
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
