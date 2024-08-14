import torch
from diffusers import FluxPipeline
import gradio as gr
import psutil
import GPUtil
import gc

class OptimizedFluxSchnellPipeline:
    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading Flux Schnell model. This might take a moment...")
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
        
        # Enable memory-efficient attention if available
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def generate(self, prompt, num_inference_steps=4, width=512, height=512, guidance_scale=3.5):
        try:
            # Clear CUDA cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()

            # Move pipeline to device
            self.pipe.to(self.device)

            # Generate the image
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale
            ).images[0]

            # Move pipeline back to CPU to free up VRAM
            self.pipe.to("cpu")

            # Clear CUDA cache and collect garbage again
            torch.cuda.empty_cache()
            gc.collect()

            return image

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

optimized_flux = OptimizedFluxSchnellPipeline()

def generate_image(prompt, num_steps, width, height, guidance_scale):
    image = optimized_flux.generate(prompt, num_inference_steps=int(num_steps), width=width, height=height, guidance_scale=guidance_scale)
    return image, get_system_info()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your image description here..."),
        gr.Slider(minimum=1, maximum=20, step=1, value=4, label="Number of Inference Steps"),
        gr.Slider(minimum=256, maximum=1024, step=32, value=512, label="Width"),
        gr.Slider(minimum=256, maximum=1024, step=32, value=512, label="Height"),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=3.5, label="Guidance Scale")
    ],
    outputs=[
        gr.Image(type="pil", label="Generated Image"),
        gr.Textbox(label="System Information")
    ],
    title="Simplified Flux Schnell Image Generator",
    description="Generate high-quality images using the Flux Schnell model with optimized VRAM usage. Enter a prompt and adjust the settings.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
