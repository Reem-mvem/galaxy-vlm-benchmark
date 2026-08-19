import os, glob, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
mp = os.environ["MODEL_LOCAL_PATH"]
proc = AutoProcessor.from_pretrained(mp)
model = AutoModelForImageTextToText.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map="auto")
ad = os.environ.get("QWEN_ADAPTER_PATH")
if ad:
    model.load_adapter(ad); print("ADAPTER:", ad, flush=True)
model.eval()
prompt = ("This galaxy image shows faint debris. Classify the debris morphology. "
          "Reply with one word: arm, stream, shell, or diffuse.")
imgs = sorted(glob.glob("/home/reem1/scratch/galaxy-vlm-benchmark/tasks/which_lsb/data/images/test/*.jpg"))[:5]
for ip in imgs:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    # thinking ON (default) so we SEE the reasoning
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[Image.open(ip).convert("RGB")], return_tensors="pt").to(model.device)
    n = inp.input_ids.shape[1]
    out = model.generate(**inp, max_new_tokens=400, do_sample=False)
    full = proc.batch_decode(out[:, n:], skip_special_tokens=True)[0]
    print(f"\n===== full output ({os.path.basename(ip)}) =====\n{full}\n", flush=True)
