# 한밭대학교 인공지능소프트웨어학과 유진경은호팀

**팀 구성**
- 20227005 진경은
- 20221055 신은호
- 20221063 이유진

## <u>Teamate</u> Project Background
- ### 필요성
  - 현재 온라인 쇼핑몰의 반품률은 약 30% 로 오프라인 매장의 10%에 비해 3배 높은 수준이다. 이러한 차이는 주로 실제 착용감을 미리 확인 할 수 없어 발생하는 사이즈 불일치 문제로 인한 것으로, 물류 비용 증가와 소비자 만족도 저하의 주요 원인이 된다.
    
- ### 기존 해결책의 문제점
  - 최근 Diffusion 모델 기반의 VITON 시스템들이 높은 품질의 합성 이미지를 생성하며 주목받고 있다. 그러나 기존의 Diffsuion 기반 VTON 시스템들은 주로 서버 환경에서의 추론을 전제로 하고 있어, 사용자의 개인 이미지가 외부 서버로 전송되는 과정에서 개인정보 유출 문제가 발생할 수 있다. 
  
## System Design
 <img width="545" height="257" alt="image" src="https://github.com/user-attachments/assets/ee22ed63-2b00-446e-b40e-73a497244634" />

 - 온디바이스

 <img width="645" height="512" alt="image" src="https://github.com/user-attachments/assets/b6f6dcbf-02f2-4e5f-8d5c-f13690aa535c" />

 - 서버 + 온디바이스

  - ### System Requirements
    - On-device 환경에서 효율적으로 실행가능한 Encoding 모델
    - On-device 환경에서 효율적으로 실행가능한 Diffusion 모델
    - 입력 조건 생성을 위한 Segementation 모델
    - 사용자 만족도를 위한 빠른 추론시간
    
## Case Study
  - ### Description
   - Viton: An image-based virtual try-on network CVPR 2018
   - CatVTON: Concatenation is all you need for Virtual Try-On with diffusion models ICLR 2025
   - MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices google 2023
  
## Conclusion
  <img width="549" height="456" alt="image" src="https://github.com/user-attachments/assets/116d57bb-9210-4051-bc22-e5963c2298b2" />
  
  - 본 프로젝트는 기존 서버 기반 VTON 시스템의 개인정보 유출 문제와 높은 추론 시간 문제를 동시에 해결하기 위해, CatVTON 모델을 On-device 환경에 최적화한 비대면 의류 착용 시스템을 구현하였다.
Jetson Orin Nano에서는 FP32 → BFP16 변환 및 DPM++ 스케줄러 적용을 통해 추론 시간 단축(30초 → 11초)을 달성하였으며, 모바일 환경인 Galaxy S25(Adreno 830)에서도 Executorch를 활용해 추론이 가능함을 확인하였다. 본 시스템은 개인 이미지 암호화 및 On-device 추론 수행으로 개인정보 유출 위험을 최소화하고, 경량화된 Diffusion 기반 VTON 모델을 통해 고품질의 가상 착용 이미지를 빠르게 생성하며, Jetson 및 모바일 디바이스 환경에서도 실시간 응용이 가능한 수준으로 최적화했다.
  
## Project Outcome
- X-Decoder 모델의 경량화를 위한 PTQ 기반 양자화 최적화 2025년 대한전자공학회 하계학술대회
