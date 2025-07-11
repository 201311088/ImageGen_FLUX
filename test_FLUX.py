import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import os
import datetime
import time

# FLUX.1 모델 ID
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_DIR = "LoRA_Flux"
OUTPUT_DIR = "generated_images"

class ImageGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("FLUX.1 Image Generator (Optimization Mode)")
        master.geometry("600x500")

        # --- GUI 구성 ---
        # 파일 선택
        file_frame = tk.Frame(master)
        file_frame.pack(pady=10, padx=20, fill="x")
        self.file_label = tk.Label(file_frame, text="No file selected.", font=("Arial", 10), fg="gray")
        self.file_label.pack(side="left", fill="x", expand=True)
        self.browse_button = tk.Button(file_frame, text="Browse Prompt File (.txt)", command=self.browse_file)
        self.browse_button.pack(side="right")
        self.selected_prompt_file = None

        # LoRA 선택
        lora_frame = tk.Frame(master)
        lora_frame.pack(pady=5, padx=20, fill="x")
        tk.Label(lora_frame, text="LoRA File:", font=("Arial", 10)).pack(side="left", padx=(0, 10))
        self.lora_files = ["None"] + self.scan_lora_files()
        self.selected_lora = tk.StringVar(master)
        self.selected_lora.set(self.lora_files[0])
        self.lora_menu = ttk.Combobox(lora_frame, textvariable=self.selected_lora, values=self.lora_files, state="readonly")
        self.lora_menu.pack(side="left", fill="x", expand=True)

        # 파라미터
        params_frame = tk.Frame(master)
        params_frame.pack(pady=10, padx=20)
        self.setup_parameters(params_frame)

        # 생성 버튼
        self.generate_button = tk.Button(master, text="이미지 생성", command=self.generate_image, font=("Arial", 12), bg="lightblue", fg="black", state=tk.DISABLED)
        self.generate_button.pack(pady=20)

        # 상태 라벨
        self.status_label = tk.Label(master, text="모델 로딩 중...", font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)

        # 모델 로드
        self.pipeline = None
        self.master.after(100, self.load_model)

    def browse_file(self):
        filepath = filedialog.askopenfilename(title="Select a prompt file", filetypes=(("Text files", "*.txt"), ("All files", "*.* wurden")))
        if filepath:
            self.selected_prompt_file = filepath
            self.file_label.config(text=os.path.basename(filepath), fg="black")
        else:
            self.selected_prompt_file = None
            self.file_label.config(text="No file selected.", fg="gray")

    def setup_parameters(self, parent_frame):
        self.param_entries = {}
        params = {
            "Inference Steps:": "20", "Guidance Scale:": "7.5",
            "Width:": "1024", "Height:": "1024",
            "Seed (-1 for random):": "-1"
        }
        r, c = 0, 0
        for label, value in params.items():
            tk.Label(parent_frame, text=label, font=("Arial", 10)).grid(row=r, column=c, sticky="w", pady=2, padx=5)
            entry = tk.Entry(parent_frame, font=("Arial", 10))
            entry.grid(row=r, column=c+1, sticky="ew", pady=2)
            entry.insert(0, value)
            self.param_entries[label] = entry
            r += 1
            if r > 2:
                r = 0
                c += 2
        parent_frame.columnconfigure(1, weight=1)
        parent_frame.columnconfigure(3, weight=1)

    def scan_lora_files(self):
        if not os.path.isdir(LORA_DIR):
            os.makedirs(LORA_DIR, exist_ok=True)
            return []
        return [f for f in os.listdir(LORA_DIR) if f.endswith((".safetensors", ".bin", ".pt", ".pth"))]

    def load_model(self):
        self.status_label.config(text=f"'{MODEL_ID}' 모델 로딩 및 컴파일 중...", fg="blue")
        self.master.update_idletasks()
        start_time = time.time()
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.pipeline = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.pipeline.to(self.device)
            
            end_time = time.time()
            self.model_load_time = end_time - start_time

            self.status_label.config(text=f"모델 로드 완료 ({self.model_load_time:.2f}s).", fg="green")
            self.generate_button.config(state=tk.NORMAL)
        except Exception as e:
            error_msg = f"모델 로드/컴파일 오류: {e}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("초기화 실패", f"{error_msg}\n콘솔 로그를 확인하세요.")
            print(error_msg)

    def generate_image(self):
        if not self.pipeline:
            messagebox.showwarning("경고", "모델이 아직 로드되지 않았습니다.")
            return

        if not self.selected_prompt_file:
            messagebox.showwarning("경고", "프롬프트 파일(.txt)을 선택해주세요.")
            return

        try:
            with open(self.selected_prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            messagebox.showerror("파일 읽기 오류", f"프롬프트 파일을 읽는 데 실패했습니다:\n{e}")
            return

        if not prompt:
            messagebox.showwarning("경고", "프롬프트 파일의 내용이 비어있습니다.")
            return

        try:
            params = {
                "num_inference_steps": int(self.param_entries["Inference Steps:"].get()),
                "guidance_scale": float(self.param_entries["Guidance Scale:"].get()),
                "width": int(self.param_entries["Width:"].get()),
                "height": int(self.param_entries["Height:"].get()),
                "seed": int(self.param_entries["Seed (-1 for random):"].get())
            }
        except ValueError:
            messagebox.showerror("입력 오류", "파라미터에 유효한 숫자 값을 입력하세요.")
            return

        self.generate_button.config(state=tk.DISABLED)
        self.status_label.config(text="이미지 생성 중...", fg="blue")
        self.master.update_idletasks()

        generation_start_time = time.time()
        selected_lora = self.selected_lora.get()
        try:
            if selected_lora != "None":
                lora_path = os.path.join(LORA_DIR, selected_lora)
                self.pipeline.load_lora_weights(lora_path)

            generator = None
            if params["seed"] != -1:
                generator = torch.Generator(device=self.device).manual_seed(params["seed"])

            output = self.pipeline(
                prompt=prompt,
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                width=params["width"],
                height=params["height"],
                generator=generator
            )
            image = output.images[0]

            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time

            # 이미지 및 분석 로그 저장
            self.save_results(image, params, generation_time, selected_lora)

        except Exception as e:
            error_msg = f"이미지 생성 오류: {e}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("생성 실패", error_msg)
            print(error_msg)
        finally:
            if selected_lora != "None":
                self.pipeline.unload_lora_weights()
            self.generate_button.config(state=tk.NORMAL)

    def save_results(self, image, params, generation_time, lora_file):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = os.path.splitext(os.path.basename(self.selected_prompt_file))[0]
        optim_info = f"_{params['num_inference_steps']}_{params['width']}_{params['height']}_{int(generation_time)}s"
        final_filename_base = f"{base_filename}{optim_info}"
        
        # 이미지 파일 경로 결정 (중복 처리)
        output_image_path = os.path.join(OUTPUT_DIR, f"{final_filename_base}.png")
        counter = 1
        while os.path.exists(output_image_path):
            output_image_path = os.path.join(OUTPUT_DIR, f"{final_filename_base}_{counter}.png")
            counter += 1
        image.save(output_image_path)

        # 분석 로그 파일 저장
        analytics_filename = f"{os.path.splitext(os.path.basename(output_image_path))[0]}_OptimizationAnalytics.txt"
        analytics_filepath = os.path.join(OUTPUT_DIR, analytics_filename)
        
        with open(analytics_filepath, 'w', encoding='utf-8') as f:
            f.write("--- Optimization Analytics ---\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {MODEL_ID}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"LoRA File: {lora_file}\n")
            f.write("\n--- Timings ---\n")
            f.write(f"Model Loading & Compilation Time: {self.model_load_time:.2f} seconds\n")
            f.write(f"Image Generation Time: {generation_time:.2f} seconds\n")
            f.write("\n--- Parameters ---\n")
            f.write(f"Resolution: {params['width']}x{params['height']}\n")
            f.write(f"Inference Steps: {params['num_inference_steps']}\n")
            f.write(f"Guidance Scale: {params['guidance_scale']}\n")
            f.write(f"Seed: {params['seed']}\n")
            f.write("\n--- Output ---\n")
            f.write(f"Saved Image: {os.path.basename(output_image_path)}\n")

        self.status_label.config(text=f"저장 완료: {os.path.basename(output_image_path)}", fg="green")
        messagebox.showinfo("성공", f"이미지와 분석 로그가 다음 경로에 저장되었습니다:\n{output_image_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()