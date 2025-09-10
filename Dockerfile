# start from a clean base image (replace <version> with the desired [release](https://github.com/runpod-workers/worker-comfyui/releases))
FROM runpod/worker-comfyui:5.4.1-base

# install custom nodes using comfy-cli
RUN comfy-node-install comfyui-kjnodes comfyui-ic-light comfyui-multigpu ComfyUI-GGUF comfyui_essentials_mb hidream_sampler

RUN comfy model download --url https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/diffusion_models/hidream_i1_full_fp16.safetensors --relative-path models/unet --filename hidream_i1_full_fp16.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_l_hidream.safetensors --relative-path models/clip --filename clip_l_hidream.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/clip_g_hidream.safetensors --relative-path models/clip --filename clip_g_hidream.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors --relative-path models/clip --filename t5xxl_fp8_e4m3fn_scaled.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors --relative-path models/clip --filename llama_3.1_8b_instruct_fp8_scaled.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/vae/ae.safetensors --relative-path models/vae --filename ae.safetensors

# Copy local static input files into the ComfyUI input directory
#COPY input/ /comfyui/input/
