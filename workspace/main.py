from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import runpod
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure model loading with RunPod optimizations
def load_model():
    logger.info("Loading Qwen2.5-VL-7B-Instruct model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="/workspace/model-cache"
    )
    logger.info("Model loaded successfully")
    return model

# Initialize model and processor once per container
logger.info("Starting model and processor initialization...")
model = load_model()
logger.info("Loading processor...")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir="/workspace/model-cache",
    revision='refs/pr/24'
)
logger.info("Processor loaded successfully")

def handler(job):
    try:
        logger.info(f"Received new job with ID: {job.get('id', 'unknown')}")
        job_input = job['input']
        
        # Validate input parameters
        logger.info("Validating input parameters...")
        if not job_input.get('message'):
            logger.error("Missing required 'message' parameter")
            return {"error": "Missing required 'message' parameter"}
        
        image_url = job_input.get('image_url', 
            "https://www.faithinnature.co.uk/cdn/shop/articles/JanuaryBlues2.jpg?v=1641828814")
        logger.info(f"Using image URL: {image_url}")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": job_input['message']}
            ]
        }]
        logger.info(f"Constructed messages with user prompt: {job_input['message']}")

        # Process inputs with error handling
        logger.info("Applying chat template...")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info("Processing vision information...")
        image_inputs, video_inputs = process_vision_info(messages)
        
        logger.info("Preparing model inputs...")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        logger.info("Inputs prepared and moved to CUDA")

        # Generate with optimized parameters
        logger.info("Starting text generation...")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        logger.info("Text generation completed")
        
        # Post-process output
        logger.info("Post-processing generated text...")
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        logger.info("Processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing job: {str(e)}", exc_info=True)
        return {"error": str(e)}

logger.info("Starting RunPod serverless handler...")
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_dict": True
})
