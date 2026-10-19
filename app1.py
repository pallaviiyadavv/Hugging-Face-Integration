import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image

# 1. Load the text-to-image model
# We use SDXL Turbo as it is exceptionally fast and high-quality
model_id = "stabilityai/sdxl-turbo"

# Check if a CUDA GPU is available, otherwise fallback to CPU
if torch.cuda.is_available():
  pipe = AutoPipelineForText2Image.from_pretrained(
      model_id, torch_dtype=torch.float16, variant="fp16"
  )
  pipe.to("cuda")
  print("Running on GPU (CUDA)")
else:
  pipe = AutoPipelineForText2Image.from_pretrained(
      model_id, torch_dtype=torch.float32
  )
  pipe.to("cpu")
  print("Running on CPU (Slow)")


# 2. Define the generation function
def generate_image(prompt):
  image = pipe(
      prompt=prompt, num_inference_steps=2, guidance_scale=0.0
  ).images[0]
  return image


# 3. Create the Gradio Interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        label="Enter your prompt",
        placeholder="A futuristic city at sunset, cinematic lighting, 8k resolution...",
    ),
    outputs=gr.Image(label="Generated Image"),
    title="Hugging Face Text-to-Image AI Generator",
    description="Type a prompt to instantly generate an AI image using Stable Diffusion XL Turbo.",
)

interface.launch(share=True)
