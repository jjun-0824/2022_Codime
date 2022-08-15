# 2022_가상시착 산학협력
---

모델 이미지와 원하는 옷을 선택하면 옷을 시착해주는, 가상시착(virtual-tryon) 관련 프로젝트입니다. 본 프로젝트는 [Style-based global appearnce flow](https://github.com/SenHe/Flow-Style-VTON#style-based-global-appearance-flow-for-virtual-try-on-cvpr-2022)와 [HR-Viton](https://github.com/sangyun884/HR-VITON) 모델을 기반으로 진행되었고, 0) 모델 환경 구축 및 성능 test, 1) 각 모델들의 Demo page를 위한 Data Pipeline 작성 및 time profiling & optimization, 2) 하의 시착 및 최신 의상 시착을 위한 추가 데이터셋 크롤링 및 fine tuning으로 프로젝트가 진행되었습니다. 

참여 일자: 2022.06.13 - 2022.08.17
참여 기관: 성균관대학교 산학협력 codime

---
## 모델링
### 1. Style-based global appearance flow(CVPR 2022)
- Train process: 
1) Semantic representation(segmentation map, keypoint pose, dense pose)과 model image, paired garments를 input으로 두어 parser-based 모델을 우선 학습 
2) Parser-free 모델을 knowledge distillation을 기반으로 하여 학습을 진행. 자세히 말해보자면, model image에 unpaired된 garments를 pretrained된 parser-based를 통해 입히고, 해당 이미지($$I_{unpaired}$$)와 paired garments를 input으로 받고 parser-free model을 통해 입힌 다음 원본 이미지($$I_{original}$$)와 생성된 이미지($$I_{paired}$$)간의 loss 값을 기반으로 parser-free model을 학습을 진행한다는 dmlal.
- Dataset: VITON(256*912)
- 특징: parser-based 모델을 기반으로 parser 없이 옷과 모델 이미지로만 학습이 가능하다는 점. knowledge distillation 기반으로 학습이 진행됨.
- 보완점 및 결론: 
- github 및 라이센스:


---
### 2. High Rebvolution Vitrual Try-On(CVPR 2022)
- Dataset: 자체 크롤링 데이터셋
 - 특징 및 로직 정리:
 - 수정:
 - Data Pipeline:
 - github 및 라이센스
 
* CIHP model을 knowledge distillation을 시키거나, 혹은 HR-VITON을 knowledge distillation을 시켜서 Input 자체를 줄이는 더 가벼운 모델을 만드는 것도 방법

---
### 3. 추가 자료

---
## 크롤링
1. 하의 
 
---
### 개발 환경
- 
- 
- 
