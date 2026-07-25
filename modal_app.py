import io
import modal
from PIL import Image

def download_model():
    from diffusers import StableDiffusionImg2ImgPipeline
    import torch

    model_id = "runwayml/stable-diffusion-v1-5"
    StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )


sd_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "accelerate", "pillow", "transformers", "diffusers")
    .run_function(download_model)
)

app = modal.App("sd-img2img-app", image=sd_image)

@app.cls(gpu="A10G")
class SDImg2ImgRunner:
    @modal.enter()
    def load_pipeline(self):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")


    @modal.method()
    def generate(self, image_bytes: bytes, prompt: str, strength: float = 0.75) -> bytes:
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_image = init_image.resize((512, 512))

        output = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=7.5
        ).images[0]

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        return buffer.getvalue()