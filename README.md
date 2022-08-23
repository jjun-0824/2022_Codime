# 2022_가상시착 산학협력

<b>* 코드는 현재 비공개로 작성 중입니다. </b>

모델 이미지와 원하는 옷을 선택하면 옷을 시착해주는, 가상시착(virtual-tryon) 관련 프로젝트입니다. 본 프로젝트는 [Style-based global appearnce flow](https://github.com/SenHe/Flow-Style-VTON#style-based-global-appearance-flow-for-virtual-try-on-cvpr-2022)와 [HR-Viton](https://github.com/sangyun884/HR-VITON) 모델을 기반으로 진행되었고, `0) 모델 환경 구축 및 성능 test`, `1) 각 모델들의 Demo page를 위한 Data Pipeline 작성 및 time profiling & optimization`, `2) 하의 시착 및 최신 의상 시착을 위한 추가 데이터셋 크롤링 및 fine tuning`으로 프로젝트가 진행되었습니다. 

`참여 일자` 2022.06.13 - 2022.08.17

`참여 기관` 성균관대학교 산학협력 codime


## 1. 가상시착(virtual-tryon) 연구 동향

가상시착은 GNN(Generative Neural Network)을 활용한 VITON이라는 연구에서 시작되었습니다. 옷을 사람의 포즈에 맞게 warping시키고, 이 옷을 사람에게 입히는 방식으로 가상시착이 진행됩니다. 이 이후에도 다양한 method를 이용한 가상시착 모델이 다수 등장하였습니다.
- Layout generation: U-Net
- Cloth-Warping: TPS(Cloth mask에 옷 이미지를 우겨넣는 방식, 옷 프린팅 부분은 왜곡 존재)/Appearance-Flow(TPS보다 독립적인 와핑 가능. 프린팅 쏠림 현상 방지 가능)
- Image Generation: U-Net/VITON-HD(spade 기법을 저해상도 이미지를 점진적으로 고해상도 이미지로 생성.)

이 이외에도 다수의 상품을 한번에 시착하는 연구, 옷의 다양한 스타일링(layering, tuckin)이 가능하게끔 하는 연구들이 진행되고 있습니다. 가상시착의 데이터셋과 관련해서는 1) 사진에서 빛의 방향이 train data와 같지 않아도 시착이 제대로 이루어지지 않기 때문에 Data augmentation을 진행해야 한다는 점 2) 치마 같은 경우는 사람의 포즈에 따라 폭이나 주름의 형태가 모두 달라지기 때문에 모양이 일그러지게 찍힐 수 있다는 점을 알게되어 가상시착을 위한 다양한 데이터셋 마련이 요함을 알 수 있었습니다.

## 2. 가상시착(virtual-tryon) 모델 

프로젝트에서 사용했던 대표적인 가상시착 모델 2가지를 보도록 하겠습니다.

### 1) Style-based global appearance flow(CVPR 2022)

![1](https://user-images.githubusercontent.com/67568001/186096273-b83d8d91-0fee-4b23-ba6f-5c0bf8cfb6fa.png)




- `Train process` Semantic representation(segmentation map, keypoint pose, dense pose)과 model image, paired garments를 input으로 두어 parser-based 모델을 우선 학습합니다. 그 이후에는 Parser-free 모델을 knowledge distillation을 기반으로 하여 학습을 진행합니다. 자세히 말해보자면, model image에 unpaired된 garments를 pretrained된 parser-based를 통해 입히고, 해당 이미지( $I_{unpaired}$ )와 paired garments를 input으로 받고 parser-free model을 통해 입힌 다음 원본 이미지( $I_{original}$ )와 생성된 이미지( $I_{paired}$ )간의 loss 값을 기반으로 parser-free model을 학습을 진행합니다.
 
- `Dataset` VITON(256*912)

- `Input` (train 시) 옷 이미지, 모델 densepose, 옷 mask 이미지, 모델 이미지, 모델 이미지 사이즈, 모델 keypoints (test 시) 모델 이미지, 옷 이미지

- `특징` parser-based 모델을 기반으로 parser 없이 옷과 모델 이미지로만 학습이 가능하다는 점과 knowledge distillation 기반으로 학습이 진행된다는 점이 있습니다.

- `보완점 및 결론` 기존 코드의 torch.nn.parallel.DistributedDataParallel 패캐지 오류로 torch.nn.DataParallel를 사용하도록 수정. 30개중 1개꼴로 시착이 잘 되었고, 생성 이미지의 해상도가 낮다는 결과로 해당 모델을 사용하지 않기로 하였습니다.

- `github 및 라이센스` [Style-based global appearnce flow](https://github.com/SenHe/Flow-Style-VTON#style-based-global-appearance-flow-for-virtual-try-on-cvpr-2022)



### 2) High Rebvolution Vitrual Try-On(CVPR 2022)
 
![SmartSelectImage_2022-08-23-16-12-25](https://user-images.githubusercontent.com/67568001/186095838-ffba690a-6f3c-4076-ad52-0a601dc808ba.png)





- `Train process` 이전 style based 모델과는 다르게 고화질 이미지 생성이 가능한 HR-VITON을 다음 모델로 선택하였습니다. 해당 모델의 구조는 크게 두 파트로 나뉩니다. (그 전 모델을 학습시키기 위한 preprocessing 과정을 거처야 합니다.) 1) try-on condition generator에서는 옷을 모델의 포즈에 맞게 warping시키고, 옷 부분과 팔을 제외한 모델의 segementation map에서 새로 착용할 옷의 segment까지 더한 새로운 모델의 segmentation map을 생성해냅니다. 2) try-on image generator에서는 모델의 segmentation map과 모델의 agnostic 이미지와 실 옷 이미지를 입력받아 최종 시착 이미지를 생성해줍니다.

- `Dataset` HR-VITON 자체 크롤링 데이터셋(1024*768)

- `Input` (train 및 test 시) 옷 이미지, 옷 masking 이미지, 모델 이미지, 모델 densepose 이미지, 모델 agnostic 이미지, 모델 agnostic parsing 이미지, 모델 keypoints

- `특징 및 로직 정리` warping과 segmentation을 같은 stage에서 처리하고, 고해상도의 이미지를 사용하여 misalignment와 pixel squeezing artifacts를 최소화한 모델입니다.

- `보완 사항` human-parsing 모델인 CIHP-PGN의 test 속도가 상당 부분을 차지합니다. 대체 human-parsing 모델인 SCHP, QANet 시도하였고, 그 결과는 하단에 참고하도록 하겠습니다.
- `model pipeline 시간 profiling 및 optimization` 상용화를 위해서는 모델을 cpu로 돌려야하지만, 시착 시간이 기하급수적으로 늘어나는 이유로 우선은 gpu로 시착을 진행하기로 하였고, 시간을 프로파일링한 결과 다음과 같은 결과가 나왔습니다. 시간을 단축하기 위해 test 시간이 비교적 짧은 schp등과 같은 human-parsing 대체모델을 사용하거나, 모델 로드 방식을 global하게 바꾸는 등의 작업을 진행하였습니다. 모델을 가볍게 만들기 위해 CIHP model을 knowledge distillation을 시키거나, 혹은 HR-VITON을 knowledge distillation을 시켜서 Input 자체를 줄이는 더 가벼운 모델을 만드는 것도 방법입니다. 모델의 성능을 높이기 위해서, 특히 그중에서도 옷의 일그러짐 현상을 해소하기 위해서 tps로 warping하는 방식을 appearnace-flow 방식으로 수정하는 대안도 존재합니다. 

- `virtual try-on의 성능평가지표` LPIPS, SSIM, FID 등이 존재합니다. 성능 평가 결과는 SSIM - 0.882, LPIPS - 0.0695 입니다.

- `github 및 라이센스`: [HR_VITON](https://github.com/sangyun884/HR-VITON)

- `Data Pipeline` Demo page에서 새로운 사용자의 이미지와 새로운 옷을 입력받아도 곧바로 시착이 가능하도록 하는 pipeline 작성합니다.

![슬라이드8](https://user-images.githubusercontent.com/67568001/184663263-61bbe886-edab-473b-b197-2f8477c9a3bb.JPG)

(1) segmentation: test_image_pgn.py, u2net_demo.py(remove_bg_by_file)

`CIHP-PGN` 옷과 모델의 parsing 이미지를 생성 [Link](https://github.com/Engineering-Course/CIHP_PGN)
위에서 언급했듯 CIHP-PGN 모델은 시착 시간이 오래 걸리기 때문에 상용화하기 어렵습니다. 이에 대한 대체 모델로 SCHP와 QANet이라는 모델을 사용하기로 하였습니다. 다만 QANet은 코드 오류로 실행시키지 못하였고, SCHP는 모델 실행속도가 80초에서 10초가량으로 줄어든 바에 비해 성능은 이전 CIHP-PGN 모델보다 좋지 못한 결과를 보여줍니다. 다만 더 많은 데이터로 학습을 진행하거나 모델의 구조를 일부 변경함으로써 더 좋은 성능을 보여주도록 수정 중에 있습니다. SCHP를 HR-VITON 학습에 적용시키기 위해 segmentation의 label의 정보가 CIHP_PGN과 동일한지 확인하는 과정을 거치고, put_palatte 함수도 일부 수정하여 보여지는 색깔도 동일하게 수정하였습니다.

(2) Keypoints openpose.pose_keypoints.py

`OpenPose` 모델의 keypoints를 생성 [Link](https://github.com/CMU-Perceptual-Computing-Lab/openpose)


(3) Denspose: apply_net_edit.py

`DensePose` 모델의 densepose 이미지 생성 [Link](https://github.com/facebookresearch/detectron2)

(4) Agnostic: test_agnostic.py, ag_dataset_demo.py

`agnostic` HR_VITON 코드에서 get_agnostic 함수 변형하여 자체 agnostic 함수 생성.


## 3. 하의 크롤링
하의 시착을 위해 SSF shopping mall에서 Input Dataset 형식에 맞게 하의 및 최신 의상 크롤링 진행하였습니다. 매장별로 크롤링 형식이 달라 매장마다의 코드를 따로 생성하였습니다.




## 4. 시착 결과 및 성능 평가



![SmartSelectImage_2022-08-23-16-16-16](https://user-images.githubusercontent.com/67568001/186095855-211830d5-ba17-438f-9a44-b0b710acef5c.png)




데모 페이지에서 모델의 이미지와 옷을 클릭하면 오른쪽 모델 사진과 같이 시착된 모델 이미지가 등장합니다.

## 5. 프로젝트 상용화 방안
사용자들은 옷의 특징과 본인들의 신체적 특징도 살릴 수 있는 가상시착 서비스를 원합니다. 다만 GAN 기반으로 이미지가 형성되는 바 이러한 세부적인 부분들은 모델이 구현하기 어려울 수 있습니다. 또한 정확한 촬영 가이드라인이 없이는 예외사항들이 많아지기 때문에 시착이 제대로 이루어지지 않을 수 있습니다. 예를 들어, 모델의 학습을 모델의 정면샷을 기반으로 진행하였기 때문에 모델의 측면샷 이미지를 넣은 경우에는 시착이 제대로 이루어지지 않을 가능성이 큽니다. 정확한 시착 이미지를 생성해내기 어렵다고 가정하였을 때 해당 가상 시착모델의 이용할 수 있는 대안은 1) 옷의 색감 조합 확인 가능 2) 스튜디오 촬영없이 모델샷 생성 가능(+ 옷만 찍으면 정해진 모델 위에 옷을 입혀줌으로써, 대형 오프라인 아울렛의 온라인 쇼핑몰 전환에 도움이 되는 측면 존재) 등이 존재합니다. 또한, 시착이 잘 안되는 경우는 어떠한 경우인지를 눈으로 확인하여 그러한 상황을 방지할 수 있는 촬영 가이드라인을 작성하여 해당 모델을 상용화할 수 있습니다.(정면 샷, 배경색 혹은 피부색과 동일한 옷 착용 금지 등)



