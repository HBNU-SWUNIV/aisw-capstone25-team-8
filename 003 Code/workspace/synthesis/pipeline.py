# # import torch
# # import time
# # import segmentation_models_pytorch as smp
# # from PIL import Image
# # import os 
# # from model.pipeline import CatVTONPipeline
# # from diffusers.image_processor import VaeImageProcessor
# # import matplotlib.pyplot as plt
# # import cv2 
# # import numpy as np 
# import sys
# import os
# import json
# from datetime import datetime
# from pathlib import Path

# def log(msg: str):
#     print(msg, file=sys.stderr, flush=True)

# def synthesis(
#         cloth_path = "/workspace/jke_capston/sythesis_input/옷1.png", 
#         person_path = "/workspace/jke_capston/sythesis_input/사람1.png",
#         height = 512,
#         width = 384,
#         erode_kernel_size = 5,
#         erode_iterations = 2,
#         output_dir = '/workspace/jke_capston/synthesis_output',
#         num_inference_steps = 15,
#         guidance_scale = 2.5
#     ):
#     try: 
#         import torch
#         import cv2
#         import numpy as np
#         from PIL import Image
#         import segmentation_models_pytorch as smp
#         from diffusers.image_processor import VaeImageProcessor
#         from model.pipeline import CatVTONPipeline
#     except ImportError as e :
#         return {
#             "success": False,
#             "error": f"파이프 라인 실행에 필요한 패키지가 없습니다. {str(e)} 패키지 다운로드 확인!"
#         }

#     try:
#         if not os.path.exists(cloth_path):
#             return {"success": False, "error": f"옷이 없는데요? 경로확인! {cloth_path}"}
#         if not os.path.exists(person_path):
#             return {"success": False, "error": f"모델이 없는데요? 경로확인! {person_path}"}
        
#         # 아웃풋 경로 생성
#         os.makedirs(output_dir, exist_ok=True)

#         # global configuration
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         attn_ckpt_version = 'vitonhd'
#         attn_ckpt = 'zhengchong/CatVTON'
#         base_ckpt = 'booksforcharlie/stable-diffusion-inpainting'
#         weight_dtype = torch.float16  # torch.float32, torch.float16, torch.bfloat16
#         skip_safety_check = True

#         seg_model_path = "/workspace/jke_capston/synthesis/seg_model.pth"
#         if not os.path.exists(seg_model_path):
#             return {
#                 "success": False,
#                 "error" : f"segmentation 모델이 없는데요? {seg_model_path}"
#             }

#         seg_model= smp.Unet("resnet50", classes=1, activation="sigmoid").to(device)
#         state_dict = torch.load(seg_model_path, map_location=device, weights_only=False)
#         seg_model.load_state_dict(state_dict)
#         seg_model = seg_model.eval()

#         img_processor = VaeImageProcessor(vae_scale_factor=8)
#         cloth = Image.open(cloth_path).convert("RGB").resize((width, height))
#         person = Image.open(person_path).convert("RGB").resize((width, height))

#         tensor_img = img_processor.preprocess(person)
#         tensor_img = tensor_img.to(device)

#         with torch.no_grad():
#             logits = seg_model(tensor_img)
#             mask = (logits > 0.5).float()

#         mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()
#         mask_pil = Image.fromarray((mask_np * 255).astype('uint8'), mode='L')
#         mask_pil = mask_pil.resize((width, height), resample=Image.NEAREST)
#         mask_np_resized = np.array(mask_pil)
        
#         kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
#         eroded_mask = cv2.erode(mask_np_resized, kernel, iterations=erode_iterations)
#         mask_pil = Image.fromarray(eroded_mask, mode='L')

#         pipeline = CatVTONPipeline(
#             attn_ckpt_version=attn_ckpt_version,
#             attn_ckpt=attn_ckpt,
#             base_ckpt=base_ckpt,
#             weight_dtype=weight_dtype,
#             device=device,
#             skip_safety_check=skip_safety_check
#         )

#         generator = torch.Generator(device=device)
        
#         results = pipeline(
#             person,
#             cloth,
#             mask_pil,
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale,
#             height=height,
#             width=width,
#             generator=generator
#         )

#         result_img = results[0]

#         # 🔹 height 기준으로 정사각형 캔버스 (512x512)
#         canvas_side = height  # 너가 원하는 케이스: 512
#         canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))  # 배경 흰색

#         # 🔹 가운데 정렬해서 붙이기 (좌우 여백 생김)
#         offset_x = (canvas_side - result_img.width) // 2
#         offset_y = (canvas_side - result_img.height) // 2
#         canvas.paste(result_img, (offset_x, offset_y))

#         # 🔹 저장
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_filename = f"synthesis_{timestamp}.png"
#         output_path = os.path.join(output_dir, output_filename)

#         canvas.save(output_path, "PNG")
        
#         return {
#             "success": True,
#             "output_path": output_path,
#             "filename": output_filename,
#             "message": "Virtual try-on synthesis completed successfully"
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": f"Synthesis failed: {str(e)}"
#         }
    
# if __name__ == "__main__":
#     if len(sys.argv) < 4:
#         print(json.dumps({
#             "success": False,
#             "error": "Usage: python pipeline.py <cloth_path> <person_path> <output_dir> [height] [width] [num_inference_steps] [guidance_scale]"
#         }))
#         sys.exit(1)

#     cloth_path = sys.argv[1]
#     person_path = sys.argv[2]
#     output_dir = sys.argv[3]

#     # CLI 인자 → 개별 변수
#     height = int(sys.argv[4]) if len(sys.argv) > 4 else 512
#     width = int(sys.argv[5]) if len(sys.argv) > 5 else 384
#     num_inference_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 15
#     guidance_scale = float(sys.argv[7]) if len(sys.argv) > 7 else 2.5

#     # ✅ 여기서 절대 height/width를 두 번 넘기지 않게 명시적인 keyword로 전달
#     result = synthesis(
#         cloth_path=cloth_path,
#         person_path=person_path,
#         output_dir=output_dir,
#         height=height,
#         width=width,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#     )

#     print(json.dumps(result))

    
#     # # global
#     # device = "cuda"
#     # attn_ckpt_version = 'vitonhd'
#     # attn_ckpt = 'zhengchong/CatVTON'
#     # base_ckpt = 'booksforcharlie/stable-diffusion-inpainting'
#     # weight_dtype = torch.float16  # torch.float32, torch.float16, torch.bfloat16
#     # skip_safety_check = True

#     # # segmentation model load
#     # seg_model = smp.Unet("resnet50", classes=1, activation='sigmoid').to(device)
#     # state_dict = torch.load("seg_model.pth", map_location=device, weights_only=False)
#     # seg_model.load_state_dict(state_dict)
#     # seg_model = seg_model.to(device)
#     # seg_model.eval()

#     # # image load and preprocessing
#     # img_processor = VaeImageProcessor(vae_scale_factor=8)
#     # cloth = Image.open(cloth_path).convert("RGB").resize((width, height))
#     # person = Image.open(person_path).convert("RGB").resize((width, height))

#     # tensor_img = img_processor.preprocess(cloth)
#     # tensor_img = tensor_img.to(device)

#     # # segmentation
#     # with torch.no_grad():
#     #     logits = seg_model(tensor_img)
#     #     mask = (logits > 0.5).float()

#     # # mask 전처리
#     # mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()
#     # mask_pil = Image.fromarray((mask_np * 255).astype('uint8'), mode='L')
#     # mask_pil = mask_pil.resize((width, height), resample=Image.NEAREST)
#     # mask_np_resized = np.array(mask_pil)
#     # kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
#     # eroded_mask = cv2.erode(mask_np_resized, kernel, iterations=erode_iterations)
#     # mask_pil = Image.fromarray(eroded_mask, mode='L')

#     # # 합성 실행 
#     # pipeline = CatVTONPipeline(
#     #     attn_ckpt_version=attn_ckpt_version,
#     #     attn_ckpt=attn_ckpt,
#     #     base_ckpt=base_ckpt,
#     #     weight_dtype=weight_dtype,
#     #     device=device,
#     #     skip_safety_check=skip_safety_check
#     # )

#     # generator = torch.Generator(device=device)

#     # results = pipeline(
#     #     person,
#     #     cloth,
#     #     mask_pil,
#     #     num_inference_steps=num_inference_steps,
#     #     guidance_scale=guidance_scale,
#     #     height=height,
#     #     width=width,
#     #     generator=generator
#     # )

#     # # 출력 디렉토리 생성
#     # os.makedirs(output_dir, exist_ok=True)

#     # # 결과 저장
#     # output_path = os.path.join(output_dir, 'result.jpg')
#     # results[0].save(output_path, 'JPEG')
#     # print(f"결과 저장 완료: {output_path}")


# # import torch
# # import time
# # import segmentation_models_pytorch as smp
# # from PIL import Image
# # import os 
# # from model.pipeline import CatVTONPipeline
# # from diffusers.image_processor import VaeImageProcessor
# # import matplotlib.pyplot as plt
# # import cv2 
# # import numpy as np 
# import sys
# import os
# import json
# from datetime import datetime
# from pathlib import Path

# def log(msg: str):
#     print(msg, file=sys.stderr, flush=True)

# def synthesis(
#         cloth_path = "/workspace/jke_capston/sythesis_input/옷1.png", 
#         person_path = "/workspace/jke_capston/sythesis_input/사람1.png",
#         height = 512,
#         width = 384,
#         erode_kernel_size = 5,
#         erode_iterations = 2,
#         output_dir = '/workspace/jke_capston/synthesis_output',
#         num_inference_steps = 15,
#         guidance_scale = 2.5
#     ):
#     try: 
#         import torch
#         import cv2
#         import numpy as np
#         from PIL import Image
#         import segmentation_models_pytorch as smp
#         from diffusers.image_processor import VaeImageProcessor
#         from catvton.pipeline import CatVTONPipeline
#     except ImportError as e :
#         return {
#             "success": False,
#             "error": f"파이프 라인 실행에 필요한 패키지가 없습니다. {str(e)} 패키지 다운로드 확인!"
#         }

#     try:
#         if not os.path.exists(cloth_path):
#             return {"success": False, "error": f"옷이 없는데요? 경로확인! {cloth_path}"}
#         if not os.path.exists(person_path):
#             return {"success": False, "error": f"모델이 없는데요? 경로확인! {person_path}"}
        
#         # 아웃풋 경로 생성
#         os.makedirs(output_dir, exist_ok=True)

#         ########################################################################
#         progress_dir = os.path.join(output_dir, "progress")
#         os.makedirs(progress_dir, exist_ok=True)
#         # 🔹 타임스탬프 미리 생성 (콜백에서 사용하기 위해)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         ########################################################################
#         def progress_callback(step, latents):
#             """매 스텝마다 latent를 이미지로 변환해서 저장"""
#             try:
#                 with torch.no_grad():
#                     # 🔹 배치가 여러 개면 첫 번째만 사용
#                     if latents.shape[0] > 1:
#                         latents_to_decode = latents[0:1]
#                     else:
#                         latents_to_decode = latents
                    
#                     # latent를 이미지로 디코딩
#                     latents_scaled = 1 / 0.18215 * latents_to_decode
#                     image = pipeline.vae.decode(latents_scaled).sample
#                     image = (image / 2 + 0.5).clamp(0, 1)
#                     image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
#                     image = (image * 255).astype(np.uint8)
                
#                 # 🔹 이미지가 위아래로 붙어있다면 위쪽 절반만 사용
#                 height = image.shape[0]
#                 if height > final_height * 1.5:  # 명백히 두 배인 경우
#                     image = image[:height//2, :, :]  # 위쪽 절반만
#                     log(f"[DEBUG] Cropped image from {height} to {image.shape[0]}")
                
#                 # PIL 이미지로 변환
#                 progress_img = Image.fromarray(image)
                
#                 # 정사각형 캔버스 (1024x1024)
#                 canvas_side = final_height
#                 canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))
#                 offset_x = (canvas_side - progress_img.width) // 2
#                 offset_y = (canvas_side - progress_img.height) // 2
#                 canvas.paste(progress_img, (offset_x, offset_y))
                
#                 # 저장
#                 progress_filename = f"progress_{timestamp}_step_{step:03d}.png"
#                 progress_path = os.path.join(progress_dir, progress_filename)
#                 canvas.save(progress_path, "PNG")
                
#                 # 🔹 stdout으로 진행상황 전송 (JSON 형식)
#                 progress_data = {
#                     "type": "progress",
#                     "step": step,
#                     "total_steps": num_inference_steps,
#                     "image_path": f"/synthesis_output/progress/{progress_filename}"
#                 }
#                 print(json.dumps(progress_data), flush=True)
                
#             except Exception as e:
#                 log(f"Progress callback error at step {step}: {str(e)}")
#         ########################################################################
#         # 🔹 내부 해상도 설정
#         seg_height = 512
#         seg_width = 384
#         final_height = 1024
#         final_width = 768

#         # global configuration
#         # "mix": "mix-48k-1024",
#         # "vitonhd": "vitonhd-16k-512",
#         # "dresscode": "dresscode-16k-512",
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         attn_ckpt_version = 'vitonhd'
#         attn_ckpt = 'zhengchong/CatVTON'
#         base_ckpt = 'booksforcharlie/stable-diffusion-inpainting'
#         weight_dtype = torch.float16  # torch.float32, torch.float16, torch.bfloat16
#         skip_safety_check = True

#         seg_model_path = "/workspace/jke_capston/synthesis/seg_model.pth"
#         if not os.path.exists(seg_model_path):
#             return {
#                 "success": False,
#                 "error" : f"segmentation 모델이 없는데요? {seg_model_path}"
#             }

#         seg_model= smp.Unet("resnet50", classes=1, activation="sigmoid").to(device)
#         state_dict = torch.load(seg_model_path, map_location=device, weights_only=False)
#         seg_model.load_state_dict(state_dict)
#         seg_model = seg_model.eval()

#         img_processor = VaeImageProcessor(vae_scale_factor=8)
        
#         # 🔹 Segmentation용: 512x384
#         cloth_seg = Image.open(cloth_path).convert("RGB").resize((seg_width, seg_height))
#         person_seg = Image.open(person_path).convert("RGB").resize((seg_width, seg_height))
        
#         # 🔹 최종 합성용: 1024x768
#         cloth = Image.open(cloth_path).convert("RGB").resize((final_width, final_height))
#         person = Image.open(person_path).convert("RGB").resize((final_width, final_height))

#         # 🔹 Segmentation (512x384)
#         tensor_img = img_processor.preprocess(person_seg)
#         tensor_img = tensor_img.to(device)

#         with torch.no_grad():
#             logits = seg_model(tensor_img)
#             mask = (logits > 0.5).float()

#         # 🔹 마스크 후처리 (512x384)
#         mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()
#         mask_pil = Image.fromarray((mask_np * 255).astype('uint8'), mode='L')
#         mask_pil = mask_pil.resize((seg_width, seg_height), resample=Image.NEAREST)
#         mask_np_resized = np.array(mask_pil)
        
#         kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
#         eroded_mask = cv2.erode(mask_np_resized, kernel, iterations=erode_iterations)
        
#         # 🔹 마스크 업스케일 (512x384 -> 1024x768)
#         mask_pil = Image.fromarray(eroded_mask, mode='L')
#         mask_pil = mask_pil.resize((final_width, final_height), resample=Image.BILINEAR)

#         # 🔹 CatVTON 파이프라인 (1024x768)
#         pipeline = CatVTONPipeline(
#             attn_ckpt_version=attn_ckpt_version,
#             attn_ckpt=attn_ckpt,
#             base_ckpt=base_ckpt,
#             weight_dtype=weight_dtype,
#             device=device,
#             skip_safety_check=skip_safety_check,
#             ########################################################################
#             progress_callback=progress_callback
#             ########################################################################
#         )

#         generator = torch.Generator(device=device)
        
#         results = pipeline(
#             person,
#             cloth,
#             mask_pil,
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale,
#             height=final_height,
#             width=final_width,
#             generator=generator
#         )

#         result_img = results[0]

#         # 🔹 정사각형 캔버스 (1024x1024)
#         canvas_side = final_height
#         canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))

#         offset_x = (canvas_side - result_img.width) // 2
#         offset_y = (canvas_side - result_img.height) // 2
#         canvas.paste(result_img, (offset_x, offset_y))

#         # 🔹 저장
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_filename = f"synthesis_{timestamp}.png"
#         output_path = os.path.join(output_dir, output_filename)

#         canvas.save(output_path, "PNG")
        
#         return {
#             "success": True,
#             "output_path": output_path,
#             "filename": output_filename,
#             "message": "Virtual try-on synthesis completed successfully"
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": f"Synthesis failed: {str(e)}"
#         }
    
# if __name__ == "__main__":
#     if len(sys.argv) < 4:
#         print(json.dumps({
#             "success": False,
#             "error": "Usage: python pipeline.py <cloth_path> <person_path> <output_dir> [height] [width] [num_inference_steps] [guidance_scale]"
#         }))
#         sys.exit(1)

#     cloth_path = sys.argv[1]
#     person_path = sys.argv[2]
#     output_dir = sys.argv[3]

#     height = int(sys.argv[4]) if len(sys.argv) > 4 else 512
#     width = int(sys.argv[5]) if len(sys.argv) > 5 else 384
#     num_inference_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 15
#     guidance_scale = float(sys.argv[7]) if len(sys.argv) > 7 else 2.5

#     result = synthesis(
#         cloth_path=cloth_path,
#         person_path=person_path,
#         output_dir=output_dir,
#         height=height,
#         width=width,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#     )

#     print(json.dumps(result))


# import torch
# import time
# import segmentation_models_pytorch as smp
# from PIL import Image
# import os 
# from model.pipeline import CatVTONPipeline
# from diffusers.image_processor import VaeImageProcessor
# import matplotlib.pyplot as plt
# import cv2 
# import numpy as np 
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import time

def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

def synthesis(
        cloth_path = "/workspace/jke_capston/sythesis_input/옷1.png", 
        person_path = "/workspace/jke_capston/sythesis_input/사람1.png",
        height = 512,
        width = 384,
        erode_kernel_size = 5,
        erode_iterations = 2,
        output_dir = '/workspace/jke_capston/synthesis_output',
        num_inference_steps = 15,
        guidance_scale = 2.5
    ):
    try: 
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        import segmentation_models_pytorch as smp
        from diffusers.image_processor import VaeImageProcessor
        from catvton.pipeline import CatVTONPipeline
        from diffusers import StableDiffusionUpscalePipeline, DPMSolverMultistepScheduler 
    except ImportError as e :
        return {
            "success": False,
            "error": f"파이프 라인 실행에 필요한 패키지가 없습니다. {str(e)} 패키지 다운로드 확인!"
        }

    try:
        if not os.path.exists(cloth_path):
            return {"success": False, "error": f"옷이 없는데요? 경로확인! {cloth_path}"}
        if not os.path.exists(person_path):
            return {"success": False, "error": f"모델이 없는데요? 경로확인! {person_path}"}
        
        # 아웃풋 경로 생성
        os.makedirs(output_dir, exist_ok=True)

        
        # 🔹 progress 저장 폴더
        progress_dir = os.path.join(output_dir, "progress")
        os.makedirs(progress_dir, exist_ok=True)
        
        # 🔹 타임스탬프 미리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔹 해상도 설정 (모두 512x384로 통일)
        final_height = 512
        final_width = 384

        # global configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        attn_ckpt_version = 'vitonhd'
        attn_ckpt = 'zhengchong/CatVTON'
        base_ckpt = 'booksforcharlie/stable-diffusion-inpainting'
        weight_dtype = torch.float16
        skip_safety_check = True

        seg_model_path = "/workspace/jke_capston/synthesis/seg_model.pth"
        if not os.path.exists(seg_model_path):
            return {
                "success": False,
                "error" : f"segmentation 모델이 없는데요? {seg_model_path}"
            }

        seg_model= smp.Unet("resnet50", classes=1, activation="sigmoid").to(device)
        state_dict = torch.load(seg_model_path, map_location=device, weights_only=False)
        seg_model.load_state_dict(state_dict)
        seg_model = seg_model.eval()

        upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=weight_dtype
        ).to(device)
        upscaler.scheduler = DPMSolverMultistepScheduler.from_config(
            upscaler.scheduler.config
        )

        img_processor = VaeImageProcessor(vae_scale_factor=8)
        
        # 🔹 모든 이미지를 512x384로 로드
        cloth = Image.open(cloth_path).convert("RGB").resize((final_width, final_height))
        person = Image.open(person_path).convert("RGB").resize((final_width, final_height))

        # 🔹 Segmentation (512x384)
        tensor_img = img_processor.preprocess(person)
        tensor_img = tensor_img.to(device)

        with torch.no_grad():
            logits = seg_model(tensor_img)
            mask = (logits > 0.5).float()

        # 🔹 마스크 후처리 (512x384)
        mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype('uint8'), mode='L')
        mask_pil = mask_pil.resize((final_width, final_height), resample=Image.NEAREST)
        mask_np_resized = np.array(mask_pil)
        
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask_np_resized, kernel, iterations=erode_iterations)
        mask_pil = Image.fromarray(eroded_mask, mode='L')

        # 🔹 progress_callback 함수 정의
        def progress_callback(step, latents):
            """매 스텝마다 latent를 이미지로 변환해서 저장"""
            try:
                with torch.no_grad():
                    log(f"[DEBUG] Step {step} latents shape: {latents.shape}")
                    
                    # 🔹 배치가 여러 개면 첫 번째만 사용
                    if latents.shape[0] > 1:
                        latents_to_decode = latents[0:1]
                    else:
                        latents_to_decode = latents
                    
                    # 🔹 높이가 2배면 절반만 사용
                    if latents_to_decode.shape[2] > 128:
                        latents_to_decode = latents_to_decode[:, :, :latents_to_decode.shape[2]//2, :]
                        log(f"[DEBUG] Cropped latents to: {latents_to_decode.shape}")
                    
                    # latent를 이미지로 디코딩
                    latents_scaled = 1 / 0.18215 * latents_to_decode
                    decoded = pipeline.vae.decode(latents_scaled).sample
                    
                    log(f"[DEBUG] Decoded image shape: {decoded.shape}")
                    
                    image = (decoded / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image = (image * 255).astype(np.uint8)
                
                # 🔹 이미지가 위아래로 붙어있다면 위쪽 절반만 사용
                img_height = image.shape[0]
                if img_height > final_height * 1.5:
                    image = image[:img_height//2, :, :]
                    log(f"[DEBUG] Cropped image from {img_height} to {image.shape[0]}")
                
                # PIL 이미지로 변환
                progress_img = Image.fromarray(image)
                
                # 정사각형 캔버스 (512x512)
                canvas_side = final_height
                canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))
                offset_x = (canvas_side - progress_img.width) // 2
                offset_y = (canvas_side - progress_img.height) // 2
                canvas.paste(progress_img, (offset_x, offset_y))
                
                # 저장
                progress_filename = f"progress_{timestamp}_step_{step:03d}.png"
                progress_path = os.path.join(progress_dir, progress_filename)
                canvas.save(progress_path, "PNG")
                
                # 🔹 stdout으로 진행상황 전송 (JSON 형식)
                progress_data = {
                    "type": "progress",
                    "step": step,
                    "total_steps": num_inference_steps,
                    "image_path": f"/synthesis_output/progress/{progress_filename}"
                }
                print(json.dumps(progress_data), flush=True)

                time.sleep(1.5)
                
            except Exception as e:
                log(f"Progress callback error at step {step}: {str(e)}")

        # 🔹 CatVTON 파이프라인 (512x384)
        pipeline = CatVTONPipeline(
            attn_ckpt_version=attn_ckpt_version,
            attn_ckpt=attn_ckpt,
            base_ckpt=base_ckpt,
            weight_dtype=weight_dtype,
            device=device,
            skip_safety_check=skip_safety_check,
            progress_callback=progress_callback
        )

        generator = torch.Generator(device=device)
        
        results = pipeline(
            person,
            cloth,
            mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=final_height,
            width=final_width,
            generator=generator
        )

        result_img = results[0]

        upscaled_result = upscaler(
            prompt="high quality, 4k, sharp details",  # 또는 "", "photo" 등
            image=result_img,
            num_inference_steps=12,   # 원하는 만큼
        )
        upscaled_img = upscaled_result.images[0].resize((768, 1024), Image.LANCZOS)
        

        # 🔹 정사각형 캔버스 (1024x1024)
        canvas_side = 1024
        canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))

        offset_x = (canvas_side - upscaled_img.width) // 2
        offset_y = (canvas_side - upscaled_img.height) // 2
        canvas.paste(upscaled_img, (offset_x, offset_y))

        # 🔹 저장
        output_filename = f"synthesis_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        canvas.save(output_path, "PNG")
        
        return {
            "success": True,
            "output_path": output_path,
            "filename": output_filename,
            "message": "Virtual try-on synthesis completed successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Synthesis failed: {str(e)}"
        }
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({
            "success": False,
            "error": "Usage: python pipeline.py <cloth_path> <person_path> <output_dir> [height] [width] [num_inference_steps] [guidance_scale]"
        }))
        sys.exit(1)

    cloth_path = sys.argv[1]
    person_path = sys.argv[2]
    output_dir = sys.argv[3]

    height = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 384
    num_inference_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    guidance_scale = float(sys.argv[7]) if len(sys.argv) > 7 else 2.5

    result = synthesis(
        cloth_path=cloth_path,
        person_path=person_path,
        output_dir=output_dir,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    print(json.dumps(result))