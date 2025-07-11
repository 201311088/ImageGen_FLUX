# FLUX Image Generation Project

This project explores high-quality image generation using the FLUX 1.D model. My primary goal is to leverage LoRA (Low-Rank Adaptation) techniques to achieve specific artistic styles and tendencies in the generated images.

## Project Structure

- `gradio_FLUX.py`: The main script for running the Gradio interface, which handles the image generation process. This is where I'm currently experimenting with converting from FP32 to FP16 due to VRAM limitations.
- `flux_onnx/`: Contains the ONNX models for FLUX, including the text encoders, transformer, unet, and VAE decoder.
- `LoRA_Flux/`: Directory for storing custom LoRA models.
- `generated_images/`: Output directory for the generated images.
- `check_cuda.py`: A utility script to verify CUDA installation.
- `requirements.txt`: Lists all necessary Python dependencies.

## Current Status & Roadmap

I am actively working on optimizing the model for lower VRAM environments, specifically by attempting to transition from FP32 to FP16 precision. This is a crucial step to make the generation process more accessible and efficient.

Looking ahead, the high-quality images generated with applied LoRA will serve as the foundational first frames for future generative video projects. This project aims to be a stepping stone towards creating compelling and stylistically consistent generative video content.

## Issues

- **Performance**: Even with an RTX 3090, generating a 512x512 image with 25 inference steps using the FLUX model takes approximately 10-15 minutes. Significant optimization is needed.
- **Gradio Stability**: Experiencing occasional, unexplainable runtime crashes when using the Gradio interface.