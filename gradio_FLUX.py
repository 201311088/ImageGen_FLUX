import gradio as gr
import asyncio
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from diffusers import DiffusionPipeline, AttnProcessor2_0
import torch
import os
import datetime
import time

# FLUX.1 모델 ID
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_DIR = "LoRA_Flux"
OUTPUT_DIR = "generated_images"

# 모델 로드 (전역 변수로 한 번만 로드)
pipeline = None

def load_model_once():
    global pipeline
    if pipeline is None:
        print("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Set backend options to avoid Triton
        torch.backends.cuda.matmul.allow_tf32 = True

        try:
            pipeline = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            pipeline.to(device)
            pipeline.set_attn_processor(AttnProcessor2_0())
            # Disable xformers to avoid Triton-related errors on Windows
            if platform.system() == "Windows":
                try:
                    pipeline.disable_xformers_memory_efficient_attention()
                    print("xformers memory efficient attention disabled for Windows.")
                except AttributeError:
                    print("Could not disable xformers, the pipeline might not support it.")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            pipeline = None # 로드 실패 시 None으로 유지
    return pipeline

def scan_lora_files():
    if not os.path.isdir(LORA_DIR):
        os.makedirs(LORA_DIR, exist_ok=True)
        return []
    return [f for f in os.listdir(LORA_DIR) if f.endswith((".safetensors", ".bin", ".pt", ".pth"))]

def generate_images(
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
    num_images: int,
    selected_lora: str
):
    global pipeline
    if pipeline is None:
        yield "모델이 아직 로드되지 않았습니다. 잠시 기다려주세요.", []
        pipeline = load_model_once()
        if pipeline is None:
            return "모델 로드에 실패했습니다. 콘솔을 확인하세요.", []

    if not prompt:
        return "프롬프트를 입력해주세요.", []

    # 상태 업데이트
    yield "이미지 생성 중... (시간이 다소 소요될 수 있습니다)", []

    generation_start_time = time.time()
    generated_image_paths = []
    
    try:
        # LoRA 적용
        if selected_lora != "None":
            lora_path = os.path.join(LORA_DIR, selected_lora)
            pipeline.load_lora_weights(lora_path)
            pipeline.to(pipeline.device) # Ensure LoRA weights are moved to the correct device
            print(f"Loaded LoRA: {selected_lora}")

        for i in range(num_images):
            current_seed = seed if seed != -1 else torch.seed()
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)

            output = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            image = output.images[0]

            # 이미지 저장 경로 설정
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # 파일명 중복 방지를 위해 현재 시간 사용
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename_base = f"generated_image_{timestamp}"
            
            # 최적화 정보는 배치 전체 시간으로 나중에 추가
            output_path = os.path.join(OUTPUT_DIR, f"{image_filename_base}_{i+1}.png")
            
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(OUTPUT_DIR, f"{image_filename_base}_{i+1}_{counter}.png")
                counter += 1

            image.save(output_path)
            generated_image_paths.append(output_path)
            yield f"이미지 {i+1}/{num_images} 생성 완료...", generated_image_paths

        generation_end_time = time.time()
        total_generation_time = generation_end_time - generation_start_time
        total_generation_minutes = int(total_generation_time / 60)

        # 이미지 파일명에 최적화 정보 추가 및 분석 로그 저장
        final_image_paths = []
        for path in generated_image_paths:
            base_name = os.path.splitext(os.path.basename(path))[0]
            optim_info = f"_{num_inference_steps}_{width}_{height}_{total_generation_minutes}min"
            new_name = f"{base_name}{optim_info}.png"
            new_path = os.path.join(OUTPUT_DIR, new_name)
            os.rename(path, new_path) # 파일명 변경
            final_image_paths.append(new_path)

        # 분석 로그 파일 저장
        analytics_filename = f"batch_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        analytics_filepath = os.path.join(OUTPUT_DIR, analytics_filename)
        
        with open(analytics_filepath, 'w', encoding='utf-8') as f:
            f.write("--- Optimization Analytics (Batch) ---\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {MODEL_ID}\n")
            f.write(f"Device: {pipeline.device}\n")
            f.write(f"LoRA File: {selected_lora}\n")
            f.write("\n--- Timings ---\n")
            f.write(f"Total Batch Generation Time: {total_generation_time:.2f} seconds ({total_generation_minutes} minutes)\n")
            f.write("\n--- Parameters ---\n")
            f.write(f"Resolution: {width}x{height}\n")
            f.write(f"Inference Steps: {num_inference_steps}\n")
            f.write(f"Guidance Scale: {guidance_scale}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Number of Images: {num_images}\n")
            f.write("\n--- Output Images ---\n")
            for img_path in final_image_paths:
                f.write(f"- {os.path.basename(img_path)}\n")

        yield f"모든 이미지 생성 및 저장 완료. 분석 로그: {os.path.basename(analytics_filepath)}", final_image_paths

    except Exception as e:
        error_msg = f"이미지 생성 오류: {e}"
        print(error_msg)
        yield error_msg, []
    finally:
        if selected_lora != "None":
            pipeline.unload_lora_weights()
            print(f"Unloaded LoRA: {selected_lora}")

# Gradio 인터페이스 정의
lora_options = ["None"] + scan_lora_files()

with gr.Blocks() as demo:
    gr.Markdown("# FLUX.1 Image Generator (Gradio Version)")
    gr.Markdown("### IAN's Optimization Testbed Float32 to float16 (VRAM 후달림)")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", lines=5, placeholder="생성할 이미지에 대한 설명을 입력하세요...")
            lora_dropdown = gr.Dropdown(label="LoRA File", choices=lora_options, value="None", interactive=True)
            num_images_input = gr.Number(label="Number of Images to Generate", value=1, minimum=1, maximum=4, step=1, interactive=True)

            with gr.Accordion("Advanced Parameters", open=False):
                inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Inference Steps", interactive=True)
                guidance_scale_slider = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, value=7.5, label="Guidance Scale", interactive=True)
                width_slider = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="Width", interactive=True)
                height_slider = gr.Slider(minimum=256, maximum=2048, step=64, value=1024, label="Height", interactive=True)
                seed_number = gr.Number(value=-1, label="Seed (-1 for random)", interactive=True)

            generate_button = gr.Button("이미지 생성", variant="primary")

        with gr.Column():
            status_output = gr.Textbox(label="Status", interactive=False)
            gallery_output = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", columns=2, height="auto")

    # 모델 사전 로드
    demo.load(load_model_once, None, None)

    generate_button.click(
        fn=generate_images,
        inputs=[
            prompt_input,
            inference_steps_slider,
            guidance_scale_slider,
            width_slider,
            height_slider,
            seed_number,
            num_images_input,
            lora_dropdown
        ],
        outputs=[status_output, gallery_output],
        api_name="generate_images"
    )

# Gradio 앱 실행 (인증 및 Public 접속 허용)
demo.launch(share=True, auth=("admin", "dkfrhflwma!1"))