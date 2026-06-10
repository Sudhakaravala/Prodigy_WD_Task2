<img width="1361" height="570" alt="Screenshot 2026-06-10 214558" src="https://github.com/user-attachments/assets/38d5298e-e0cb-43fd-aa3e-96bedcbf114e" />

It is code for the Image Generation :
import torch
from diffusers import StableDiffusionPipeline

# 1. Load the pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# 2. Move processing to GPU
pipe = pipe.to("cuda")
print("Model is ready!")

# 3. Enter your text description here
prompt = "A beautiful futuristic city with flying cars, neon lights, cyber-punk style, 4k resolution"

# 4. Generate the image
print("Generating your image, please wait...")
image = pipe(prompt).images[0]

# 5. Display the image directly in your notebook
display(image)

The output image for the code:
<img width="1047" height="707" alt="Screenshot 2026-06-10 214620" src="https://github.com/user-attachments/assets/9f6a2e6b-97c6-448b-b189-4c7ed1c9d89e" />
