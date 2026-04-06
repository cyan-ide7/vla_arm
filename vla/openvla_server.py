import torch
import io
import base64
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForVision2Seq, AutoProcessor
import uvicorn

app = FastAPI()

print("[VLA] Loading OpenVLA model...")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model     = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype       = torch.bfloat16,
    low_cpu_mem_usage = True,
    trust_remote_code = True
).to(DEVICE)
print(f"[VLA] Model ready on {DEVICE}")


class InferenceRequest(BaseModel):
    image_b64   : str   # base64 encoded RGB image
    instruction : str


@app.post("/predict")
def predict(req: InferenceRequest):
    # Decode image
    img_bytes = base64.b64decode(req.image_b64)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    prompt = f"In: What action should the robot take to {req.instruction}?\nOut:"

    inputs = processor(prompt, pil_image).to(DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key = "bridge_orig",
            do_sample  = False
        )

    return {
        "joints" : action[:6].tolist(),
        "gripper": float(action[6])
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)