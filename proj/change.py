import os

def rename_images_in_range(directory, old_prefix, new_prefix, old_suffix, new_suffix, start_index=126, end_index=149):
    """
    디렉토리 내의 이미지 파일 이름을 지정된 범위에 따라 변경합니다.
    
    Args:
        directory (str): 이미지 파일이 위치한 디렉토리 경로
        old_prefix (str): 기존 파일 이름의 접두사
        new_prefix (str): 새 파일 이름에 사용할 접두사
        old_suffix (str): 기존 파일 이름의 접미사
        new_suffix (str): 새 파일 이름에 사용할 접미사
        start_index (int): 범위의 시작 인덱스 (기본값: 101)
        end_index (int): 범위의 끝 인덱스 (기본값: 149)
    """
    files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg') and f.startswith(old_prefix)]
    files.sort()  # 파일을 정렬하여 순차적으로 이름을 변경

    index_range = list(range(start_index, end_index + 1))
    if len(files) > len(index_range):
        print("Warning: More files than the specified index range. Some files will not be renamed.")
        files = files[:len(index_range)]

    for i, filename in enumerate(files):
        new_index = index_range[i]
        parts = filename.split('_')
        if len(parts) >= 3 and parts[-1].startswith(old_suffix):
            base_number = new_index
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, f"{new_prefix}_{base_number:03d}_{new_suffix}.jpg")
            os.rename(src, dst)
            print(f"Renamed {src} to {dst}")

# 사용 예시
directory = 'D:/code/mmp/w11/n'
old_prefix = 'tr_image'
new_prefix = 'ts_image'
old_suffix = '003'
new_suffix = '003'
rename_images_in_range(directory, old_prefix, new_prefix, old_suffix, new_suffix)
