from fastapi import FastAPI, HTTPException
import modal

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Backend is live"}

@app.post("/generate")
def generate_text(prompt: str):
    try:
        ModelRunner = modal.Cls.from_name("full-application-2", "ModalRunner")
        model = ModelRunner()

        result = model.predict_and_save.remote(prompt=prompt)

        return {
            "status": "success",
            "output": result["text"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





























# import modal

# def run_backend():
#     ModelRunner = modal.Cls.from_name("full-application-2", "ModalRunner")
#     model = ModelRunner()

#     prompt_text = "Artificial Intelligence will"
#     response = model.predict_and_save.remote(prompt=prompt_text)
#     print(response)

#     saved_text = model.read_saved_output.remote()
#     print(saved_text)


# if __name__ == "__main__":
#     run_backend()