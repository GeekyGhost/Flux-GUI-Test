import torch
from diffusers import FluxPipeline
import gradio as gr
import psutil
import GPUtil
import gc
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file as load_sft

class OptimizedFluxPipeline:
    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load models in CPU memory
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
        self.t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", torch_dtype=self.dtype)
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dtype)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Enable memory-efficient attention if available
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_vae_slicing()

    def prepare_inputs(self, prompt, num_images=1):
        t5_input = self.t5_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        clip_input = self.clip_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
        
        return {
            "t5_input_ids": t5_input.input_ids.repeat(num_images, 1),
            "t5_attention_mask": t5_input.attention_mask.repeat(num_images, 1),
            "clip_input_ids": clip_input.input_ids.repeat(num_images, 1),
            "clip_attention_mask": clip_input.attention_mask.repeat(num_images, 1),
        }

    @torch.inference_mode()
    def generate(self, prompt, num_images=1, num_inference_steps=4, width=512, height=512, guidance_scale=3.5):
        try:
            # Clear CUDA cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()

            # Prepare inputs
            inputs = self.prepare_inputs(prompt, num_images)

            # Move models to GPU for inference
            self.pipe.to(self.device)
            self.t5_model.to(self.device)
            self.clip_model.to(self.device)

            # Generate embeddings
            t5_embeds = self.t5_model(input_ids=inputs["t5_input_ids"].to(self.device), attention_mask=inputs["t5_attention_mask"].to(self.device)).last_hidden_state
            clip_embeds = self.clip_model(input_ids=inputs["clip_input_ids"].to(self.device), attention_mask=inputs["clip_attention_mask"].to(self.device)).last_hidden_state

            # Generate image
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                images = self.pipe(
                    prompt_embeds=t5_embeds,
                    pooled_prompt_embeds=clip_embeds,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    output_type="pil"
                ).images

            # Move models back to CPU
            self.pipe.to("cpu")
            self.t5_model.to("cpu")
            self.clip_model.to("cpu")

            # Clear CUDA cache and collect garbage again
            torch.cuda.empty_cache()
            gc.collect()

            return images[0]
        except Exception as e:
            print(f"Error during image generation: {str(e)}")
            return None

def get_system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        gpu_temp = gpu.temperature
        gpu_load = gpu.load * 100
        gpu_memory = gpu.memoryUtil * 100
    else:
        gpu_temp = gpu_load = gpu_memory = 0
    
    return f"CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}% | GPU Load: {gpu_load:.1f}% | GPU Temp: {gpu_temp:.1f}°C | GPU Mem: {gpu_memory:.1f}%"

optimized_flux = OptimizedFluxPipeline()

def generate_image(prompt, num_steps, width, height, guidance_scale):
    image = optimized_flux.generate(prompt, num_inference_steps=int(num_steps), width=width, height=height, guidance_scale=guidance_scale)
    return image, get_system_info()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your image description here..."),
        gr.Slider(minimum=1, maximum=50, step=1, value=4, label="Number of Inference Steps"),
        gr.Slider(minimum=256, maximum=1024, step=32, value=512, label="Width"),
        gr.Slider(minimum=256, maximum=1024, step=32, value=512, label="Height"),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=3.5, label="Guidance Scale")
    ],
    outputs=[
        gr.Image(type="pil", label="Generated Image"),
        gr.Textbox(label="System Information")
    ],
    title="Optimized Flux Image Generator (Low VRAM Usage)",
    description="Generate high-quality images using the Flux model with optimized VRAM usage. Enter a prompt and adjust the settings.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()