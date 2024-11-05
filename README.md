# MMP
## multi media programing(Sound &amp; OpenCV)



# 스터디 개요

## 목차

### 1. Sound Processing (사운드 처리)

#### 1.1 이동 평균 필터
- **목적**: 이동 평균 필터를 저역통과 필터(LPF)로 사용하여 음성 신호의 고주파 성분을 줄여 음질을 개선하는 실험을 수행했습니다.
- **결과**: 필터의 탭(tap) 수를 증가시켜 고주파 성분이 제거되는 과정을 확인하였습니다. 탭 수가 커질수록 음성이 더 뭉개지는 효과가 나타났습니다.

#### 1.2 윈도우 방법 (직사각형, Hamming)
- **설명**: 직사각형 및 Hamming 윈도우를 적용하여 필터링 효과를 비교하였습니다.
- **결과**: Hamming 윈도우를 적용한 필터가 누설이 적고, 더 부드러운 소리를 제공함을 확인했습니다.

#### 1.3 고역통과 (HPF), 대역통과 (BPF), 대역저지 필터 (BSF)
- **설명**: HPF는 고주파 소리(예: 자음) 강화, BPF는 특정 대역 통과, BSF는 주파수 대역의 일부 제거에 사용되었습니다.
- **결과**: HPF로 고주파 자음의 선명도를 높였고, BSF로 특정 모음의 음질 변화를 관찰했습니다.

#### 1.4 사운드 렌더링 (방향 인식 실험)
- **설명**: 저주파 대역에서 ITD (Interaural Time Difference)와 고주파 대역에서 IID (Interaural Level Difference)를 이용해 방향성을 실험했습니다.
- **결과**: 저주파에서는 ITD가, 고주파에서는 IID가 더 큰 영향력을 미쳤습니다. 이를 바탕으로 방향 인식 정확도 게임을 설계해 오차 및 적중도를 조사하였습니다.

#### 1.5 음성 인식 및 변환 (ASR, TTS)
- **ASR (Automatic Speech Recognition)**: Google의 `speech_recognition` 모듈을 사용하여 음성을 텍스트로 변환하는 실험을 수행했습니다. 억양이나 운율에 따라 정확도가 달라짐을 확인했습니다.
- **TTS (Text-to-Speech)**: `gTTS` 모듈을 활용해 텍스트를 음성으로 변환하는 실험을 진행했습니다. 문장 부호에 따른 미묘한 억양 차이를 관찰했습니다.
- **결론**: ASR과 TTS에서 억양을 완벽히 구분하기는 어렵지만, 일부 문장 부호에 따른 음성의 변화가 가능함을 확인했습니다.

---

### 2. OpenCV (영상 처리)

#### 2.1 영상 필터링 (저역통과 LPF, 고역통과 HPF)
- **설명**: 세 가지 이미지를 대상으로 LPF 및 HPF를 적용해 각각의 주파수 성분을 조절했습니다.
- **결과**: 고주파 성분이 많은 이미지에서 LPF를 적용할수록 이미지가 부드럽게 변했고, HPF를 적용해 엣지와 세부적인 부분을 강조했습니다.

#### 2.2 Fourier 변환과 필터링
- **목적**: Fourier 변환을 통해 이미지의 주파수 성분을 분석하고, 다양한 패턴을 갖는 이미지의 주파수 특성을 연구했습니다.
- **결과**: 일정한 패턴(격자, 물결 등)과 불규칙한 패턴(곡선 등)에 대해 Fourier 변환을 수행해, 주파수 영역에서의 패턴 차이를 시각적으로 확인했습니다.

#### 2.3 화소 단위 처리 & 화질 개선
- **설명**: 크로마키(배경 변경) 기법을 적용해 배경을 다른 이미지로 대체하는 실험을 수행했습니다.
- **추가 실험**: R, G, B 채널을 개별적으로 조정해 화질 개선을 수행하고 히스토그램 평활화를 통해 색상 및 명암을 조정하였습니다.

#### 2.4 허프 변환 (Hough Transform)
- **목적**: Hough 변환을 활용해 직선 및 원형 검출을 실험했습니다.
- **결과**: 야구장의 파울 라인 등 특정 직선을 검출하는 데 성공하였으며, 다양한 threshold와 Canny 필터 조정으로 최적의 선 검출 파라미터를 찾았습니다.
