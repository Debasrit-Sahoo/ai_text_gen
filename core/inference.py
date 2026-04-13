from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import asyncio
from config import settings
from huggingface_hub import login

login(token=settings.huggingface_token)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
model = AutoModelForCausalLM.from_pretrained(
    settings.model_name,
    quantization_config=quantization_config,
    device_map={"": 0} #used balanced or auto for prod with actual gpus
)

inference_lock = asyncio.Lock()

def run_inference(message: str) -> str:
    try:
        prompt = f"<s>[INST] {message} [/INST]"
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        )
        inputs = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=settings.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
                repetition_penalty=1.1,
            )

        response_ids = output_ids[:, inputs.shape[-1]:]
        return tokenizer.decode(response_ids[0], skip_special_tokens=True)

    except Exception as e:
        raise RuntimeError(f"inference failed: {str(e)}")