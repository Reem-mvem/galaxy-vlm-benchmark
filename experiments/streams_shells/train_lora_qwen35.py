"""LoRA fine-tune Qwen3.5-4B (vision-language, dense) on one galaxy task (bf16, no
quantization). Adapted from the Qwen2-VL trainer — ONLY the model/processor classes
and the image-token source changed (AutoModelForImageTextToText + AutoProcessor,
image_token_id read from config = 248056). Everything else (env-driven interface,
collator, LoRA, weighted trainer) is identical, so it slots into the same pipeline.

Required env: TRAIN_SAMPLE_CSV, ADAPTER_OUT, TASK_PROMPT, [ANSWER_YES_LABEL]
Optional: IMAGES_DIR, TARGET_MODULES(all-linear), EPOCHS, BATCH_SIZE, GRAD_ACCUM,
          POS_WEIGHT, LORA_R, MODEL_LOCAL_PATH
"""
import os
import argparse
import pandas as pd
from PIL import Image

TRAIN_CSV = os.environ["TRAIN_SAMPLE_CSV"]
IMAGES_DIR = os.environ.get("IMAGES_DIR", "")
OUTPUT_DIR = os.environ["ADAPTER_OUT"]

_tm = os.environ.get("TARGET_MODULES", "all-linear")
TARGET_MODULES = _tm if _tm == "all-linear" else [m.strip() for m in _tm.split(",")]
EPOCHS = float(os.environ.get("EPOCHS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LORA_R = int(os.environ.get("LORA_R", "8"))              # rank (bump to push capacity)
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "1.0"))

PROMPT = os.environ["TASK_PROMPT"]
ANSWER_YES_LABEL = os.environ.get("ANSWER_YES_LABEL")


def load_examples():
    df = pd.read_csv(TRAIN_CSV)
    exs = []
    for _, r in df.iterrows():
        ans = ("yes" if r["label_name"] == ANSWER_YES_LABEL else "no") \
            if ANSWER_YES_LABEL else str(r["label_name"])
        p = r["file_loc"] if ("file_loc" in df.columns
                              and isinstance(r["file_loc"], str)
                              and os.path.exists(r["file_loc"])) else \
            f"{IMAGES_DIR}/{r['id_str']}.jpg"
        exs.append({"image_path": p, "answer": ans})
    return exs


def build_collate_fn(processor, image_token_id):
    pad_id = processor.tokenizer.pad_token_id

    def collate(examples):
        texts, images = [], []
        for ex in examples:
            msgs = [
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ex["answer"]}]},
            ]
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, enable_thinking=False))  # direct answer, no reasoning (match 7B)
            images.append([Image.open(ex["image_path"]).convert("RGB")])
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100              # don't train on padding
        labels[labels == image_token_id] = -100      # don't train on image placeholder tokens
        batch["labels"] = labels
        if POS_WEIGHT != 1.0:
            import torch as _t
            batch["class_weights"] = _t.tensor(
                [POS_WEIGHT if ex["answer"] == "yes" else 1.0 for ex in examples])
        return batch

    return collate


def make_weighted_trainer_cls():
    import torch
    import torch.nn as nn
    from trl import SFTTrainer

    class WeightedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("class_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            tok = ce(shift_logits.view(-1, shift_logits.size(-1)),
                     shift_labels.view(-1)).view(shift_labels.shape)
            mask = (shift_labels != -100).float()
            ex_loss = (tok * mask).sum(1) / mask.sum(1).clamp(min=1)
            w = weights.to(ex_loss.device).float()
            loss = (ex_loss * w).sum() / w.sum()
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def main(dryrun):
    from transformers import AutoProcessor, AutoConfig
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig

    model_path = os.environ.get("MODEL_LOCAL_PATH",
                                "/home/reem1/scratch/finetuning_Qwen3.5-VL-4B/Qwen3.5-4B")
    processor = AutoProcessor.from_pretrained(model_path)
    cfg = AutoConfig.from_pretrained(model_path)
    image_token_id = getattr(cfg, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"image_token_id = {image_token_id}")

    exs = load_examples()
    ans_counts = pd.Series([e["answer"] for e in exs]).value_counts().to_dict()
    print(f"{len(exs)} training examples; answer balance: {ans_counts}")
    print(f"batch: micro={BATCH_SIZE} x accum={GRAD_ACCUM} = effective {BATCH_SIZE * GRAD_ACCUM}; "
          f"epochs={EPOCHS}; lora_r={LORA_R}")
    assert all(os.path.exists(e["image_path"]) for e in exs[:5]), "image path check failed"
    ds = Dataset.from_list(exs)

    collate = build_collate_fn(processor, image_token_id)
    b = collate(exs[:2])
    print(f"collate OK: keys={list(b.keys())} input_ids={tuple(b['input_ids'].shape)} "
          f"labels={tuple(b['labels'].shape)} supervised_tokens={(b['labels'] != -100).sum().item()}")

    print(f"target_modules = {TARGET_MODULES}")
    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM")

    args = SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=float(os.environ.get("LEARNING_RATE", "2e-4")),
        warmup_ratio=0.03, logging_steps=5, save_strategy="no",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3, report_to="none", remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True})
    print("SFTConfig + LoraConfig constructed OK")

    if dryrun:
        print("DRYRUN_OK"); return

    import torch
    from transformers import AutoModelForImageTextToText
    from trl import SFTTrainer

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    print("model class:", type(model).__name__)

    TrainerCls = make_weighted_trainer_cls() if POS_WEIGHT != 1.0 else SFTTrainer
    print(f"POS_WEIGHT = {POS_WEIGHT} -> {TrainerCls.__name__}")
    trainer = TrainerCls(
        model=model, args=args, train_dataset=ds,
        data_collator=collate, peft_config=peft_config, processing_class=processor)

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    hit = sorted({n.split(".lora_")[0].split(".")[-1]
                  for n, _ in trainer.model.named_parameters() if "lora_" in n})
    print("LoRA attached to module types:", hit)

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("TRAIN_DONE ->", OUTPUT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dryrun", action="store_true")
    main(p.parse_args().dryrun)
