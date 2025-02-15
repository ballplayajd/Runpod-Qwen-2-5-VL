from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import runpod
import torch

# Configure model loading with RunPod optimizations
def load_model():
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="/workspace/model-cache"
    )

# Initialize model and processor once per container
model = load_model()
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir="/workspace/model-cache",
    revision='refs/pr/24'
)

def handler(job):
    try:
        job_input = job['input']
        
        # Validate input parameters
        if not job_input.get('message'):
            return {"error": "Missing required 'message' parameter"}
        
        image_url = job_input.get('image_url', 
            "https://www.faithinnature.co.uk/cdn/shop/articles/JanuaryBlues2.jpg?v=1641828814")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": job_input['message']}
            ]
        }]

        # Process inputs with error handling
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate with optimized parameters
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Post-process output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_dict": True
})
