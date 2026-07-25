import modal 
import os

def download_hf_modal():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    print(f"Downloading {model_name} into ocntainer image...")

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)

model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers")
    .run_function(download_hf_modal)
)

app = modal.App("full-application-2", image=model_image)

volume = modal.Volume.from_name("full-application-2-volume", create_if_missing=True)
volume_path = "/my_vol"

@app.cls(gpu="T4", volumes={volume_path: volume})
class ModalRunner:
    @modal.enter()
    def load_model(self):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model="gpt2")


    @modal.method()
    def predict_and_save(self, prompt: str) -> dict:
        res = self.pipe(prompt, max_length=50, num_return_sequences=1)
        text = res[0]["generated_text"]

        output_file = os.path.join(volume_path, "latest_prompt_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        volume.commit()

        return {
            "text": text,
            "volume": output_file   
        }

    @modal.method()
    def read_saved_output(self) -> str:
        output_file = os.path.join(volume_path, "latest_prompt_output.txt")
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                return f.read()
        return "No file foundon volume yet."