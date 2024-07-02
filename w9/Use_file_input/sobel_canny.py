import cv2
import numpy as np

width = 320
height = 240

# 이미지 파일 경로 직접 설정
image_path = 'D:/code/mmp/w9/data/test.jpg'  # 이미지 경로를 여기에 입력하세요

# 이미지 파일 읽기 및 크기 조정
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
if frame is None:
    print("Error: Image could not be read. Check the path.")
    exit()
frame = cv2.resize(frame, (width, height))  # 이미지를 지정된 width와 height로 조정

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
canny = cv2.Canny(gray, 250, 310)
cframe = np.hstack((gray, sobel, canny))
cv2.imshow('High-pass filtered results', cframe)

# 'q' 키를 누르면 종료
key = cv2.waitKey(0)  # 대기 시간을 무한대로 설정하여 사용자가 키를 누를 때까지 기다립니다.
if key == ord('q'):
    cv2.destroyAllWindows()


# Canny 대 Sobel의 비교 및 보완점:
# Sobel 필터는 엣지 검출에 사용되는 기본적인 기법 중 하나로, 이미지의 수평 및 수직 방향의 그래디언트 강도를 이용해 엣지를 감지합니다. 그러나 Sobel 필터는 노이즈에 취약하고 엣지 방향의 정밀한 판단이 부족할 수 있습니다.

# Canny 엣지 검출기는 Sobel의 단점을 보완하기 위해 설계되었습니다. Canny는 다음과 같은 방법으로 Sobel의 한계를 개선합니다:

# 노이즈 감소: Canny는 가우시안 필터를 사용하여 이미지의 노이즈를 줄이는 전처리 단계를 거칩니다. 이는 엣지 검출의 정확도를 높이는 데 도움을 줍니다.
# 엣지 그래디언트 강도 계산: Canny는 Sobel과 유사하게 그래디언트 강도를 계산하지만, 이후 단계에서 더 정교하게 처리합니다.
# 비최대 억제 (Non-maximum Suppression): 이 과정에서 Canny는 픽셀 주변에서 최대 그래디언트 값을 가지지 않는 픽셀을 제거하여 엣지가 얇아집니다.
# 이력 임곗값 처리 (Hysteresis Thresholding): Canny는 두 개의 임계값 (낮은 임계값과 높은 임계값)을 사용하여 진정한 엣지와 노이즈를 구분합니다. 이 단계에서 낮은 임계값 이상이면서 높은 임계값에 연결된 엣지만을 최종 엣지로 선택합니다.
# 이러한 보완점은 Canny가 Sobel 대비 더 정확하고 신뢰할 수 있는 엣지 검출 결과를 제공하는 데 기여합니다.