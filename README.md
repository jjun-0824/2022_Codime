# 2022_가상시착 산학협력
---

모델 이미지와 원하는 옷을 선택하면 옷을 시착해주는, 가상시착(virtual-tryon) 관련 프로젝트입니다. 본 프로젝트는 [Style-based global appearnce flow](https://github.com/SenHe/Flow-Style-VTON#style-based-global-appearance-flow-for-virtual-try-on-cvpr-2022)와 [HR-Viton](https://github.com/sangyun884/HR-VITON) 모델을 기반으로 진행되었고, `0) 모델 환경 구축 및 성능 test`, `1) 각 모델들의 Demo page를 위한 Data Pipeline 작성 및 time profiling & optimization`, `2) 하의 시착 및 최신 의상 시착을 위한 추가 데이터셋 크롤링 및 fine tuning`으로 프로젝트가 진행되었습니다. 

`참여 일자` 2022.06.13 - 2022.08.17

`참여 기관` 성균관대학교 산학협력 codime

---
## 모델링
### 1. Style-based global appearance flow(CVPR 2022)
- <b>Train process</b>
1) Semantic representation(segmentation map, keypoint pose, dense pose)과 model image, paired garments를 input으로 두어 parser-based 모델을 우선 학습 
2) Parser-free 모델을 knowledge distillation을 기반으로 하여 학습을 진행. 자세히 말해보자면, model image에 unpaired된 garments를 pretrained된 parser-based를 통해 입히고, 해당 이미지( $I_{unpaired}$ )와 paired garments를 input으로 받고 parser-free model을 통해 입힌 다음 원본 이미지( $I_{original}$ )와 생성된 이미지( $I_{paired}$ )간의 loss 값을 기반으로 parser-free model을 학습을 진행한다는 dmlal.
- <b>Dataset</b>: VITON(256*912)
- <b>특징</b>: parser-based 모델을 기반으로 parser 없이 옷과 모델 이미지로만 학습이 가능하다는 점. knowledge distillation 기반으로 학습이 진행됨.
- <b>보완점 및 결론</b>: 기존 코드의 `torch.nn.parallel.DistributedDataParallel` 패캐지 오류로 `torch.nn.DataParallel`를 사용하도록 수정. 30개중 1개꼴로 시착이 잘 되었고, 생성 이미지의 해상도가 낮은 편.
- <b>github 및 라이센스</b>: [Style-based global appearnce flow](https://github.com/SenHe/Flow-Style-VTON#style-based-global-appearance-flow-for-virtual-try-on-cvpr-2022)
- <b>code</b>: `Style-based_global_appearance_flow`


### 2. High Rebvolution Vitrual Try-On(CVPR 2022)
- <b>Dataset</b>: HR-VITON 자체 크롤링 데이터셋(1024*768)
- <b>특징 및 로직 정리</b>: warping과 segmentation을 같은 stage에서 처리하고, 고해상도의 이미지를 사용하여 misalignment와 pixel squeezing artifacts를 최소화한 모델.
- <b>보완 사항</b>: human-parsing 모델인 CIHP-PGN의 test 속도가 상당부분을 차지. 다른 human-parsing 모델인 SCHP, QANet 시도 중. 
- <b>Data Pipeline</b>: Demo page에서 새로운 사용자의 이미지와 새로운 옷을 입력받아도 곧바로 시착이 가능하도록 하는 pipeline 작성 
![슬라이드8](https://user-images.githubusercontent.com/67568001/184663263-61bbe886-edab-473b-b197-2f8477c9a3bb.JPG)
  1) CIHP-PGN: 옷과 모델의 parsing 이미지를 생성 [Link](https://github.com/Engineering-Course/CIHP_PGN)
  2) OpenPose: 모델의 keypoints를 생성 [Link](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  3) DensePose: 모델의 densepose 이미지 생성 [Link](https://github.com/facebookresearch/detectron2)
  4) agnostic: HR_VITON 코드에서 `get_agnostic` 함수 변형하여 자체 agnostic 함수 생성.
- <b>github 및 라이센스</b>: [HR_VITON](https://github.com/sangyun884/HR-VITON)
- <b>code</b>: `HR_VITON`
 
* CIHP model을 knowledge distillation을 시키거나, 혹은 HR-VITON을 knowledge distillation을 시켜서 Input 자체를 줄이는 더 가벼운 모델을 만드는 것도 방법


### 3. 추가 자료
- Layout generation: U-Net
- Cloth-Warping: TPS(Cloth mask에 옷 이미지를 우겨넣는 방식이라 옷 프린팅 부분은 왜곡 존재)/Appearance-Flow(TPS보다 독립적인 와핑 가능. 프린팅 쏠림 현상 없음)
- Image Generation: U-Net/VITON-HD(spader기법을 저해상도 이미지를 점진적으로 고해상도 이미지로 생성.)

---
## 크롤링
- SSF shopping mall에서 Input Dataset 형식에 맞게 하의 및 최신 의상 크롤링 진행

---
## 비즈니스적 측면 
정확한 시착 이미지를 생성해내기 어렵다고 가정하였을 때...
- 옷의 색감 조합 확인 가능
- 스튜디오 촬영없이 모델샷 생성 가능(+ 옷만 찍으면 정해진 모델 위에 옷을 입혀줌으로써, 대형 오프라인 아울렛의 온라인 쇼핑몰 전환에 도움이 되는 측면 존재)

---
## 동향 및 한계점
- 상.하의 동시착용
- 착용방법에 따른(tuck-in, layering) 다른 이미지 생성
- 고해상도 이미지 생성
- 한계: 사용자의 신체적 특징 혹은 옷의 특징이 정확히 반영된 모델 시착 이미지는 현 단계에서 생성하기 어려움.

---
* 코드는 현재 비공개로 작성 중입니다.


