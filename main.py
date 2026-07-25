import modal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response

app = FastAPI(title="Stable Diffusion image 2 image model api")

@app.get("/")
def health_check():
    return {
        "status": "IMG2IMG Backend is running"
    }

@app.post('/transform')
async def transform_image(prompt: str = Form(...), strength: float = Form(0.75), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        ModelRunner = modal.Cls.from_name("sd-img2img-app", "SDImg2ImgRunner")
        runner = ModelRunner()

        result_bytes = runner.generate.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            strength=strength
        )

        return Response(content=result_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))