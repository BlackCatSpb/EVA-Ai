import os
import sys
import traceback

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
except Exception as e:
    print(f"[ERR] Failed to import dependencies: {e}")
    sys.exit(1)


def main():
    print("[START] quick_gen_test.py")
    model_path = os.environ.get(
        "COGNIFLEX_DEFAULT_TEXT_GEN",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "cogniflex", "mlearning", "cogniflex_models", "rugpt_small")),
    )
    print(f"[INFO] model_path = {model_path}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"[INFO] cuda_available = {use_cuda}")
    if use_cuda:
        try:
            print(f"[INFO] device_name = {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print("[INFO] tokenizer loaded")
    except Exception as e:
        print(f"[ERR] Failed to load tokenizer: {e}")
        traceback.print_exc()
        sys.exit(2)

    try:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        torch_dtype = torch.float16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        print(f"[INFO] model loaded: {getattr(cfg, 'model_type', 'unknown')} -> dtype={torch_dtype}")
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(3)

    prompt = "Привет! Расскажи кратко о себе."
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.generate(**inputs, **gen_kwargs)
            else:
                out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("\n=== Generated Text (direct HF) ===\n")
        print(text)
    except Exception as e:
        print(f"[ERR] Generation failed: {e}")
        traceback.print_exc()
        sys.exit(4)

    print("[DONE] quick_gen_test.py")


if __name__ == "__main__":
    main()
