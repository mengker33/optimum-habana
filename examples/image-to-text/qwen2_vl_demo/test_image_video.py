import time
from PIL import Image
import requests
from transformers import AutoProcessor
from optimum.habana.transformers.models import GaudiQwen2VLForConditionalGeneration
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from qwen_vl_utils import process_vision_info
import torch

adapt_transformers_to_gaudi()

bs=8

model = GaudiQwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = model.to("hpu")
model.to(torch.bfloat16)
wrap_in_hpu_graph(model)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


messages0 = [
    {"role": "user", "content": [
        {"type": "text", 
         "text": "What is deep learning?"},
        ]
    },
]

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "Beijing.jpeg" 
            },
            {"type": "text", "text": "Describe the image."},
        ],
    }
]

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "cherry_blossom.jpg" 
            },
            {"type": "text", "text": "Describe the image."},
        ],
    }
]

messages3 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "space_woaudio.mp4",
                 },
                {"type": "text", "text": "Describe this video."},
            ],
        }
]

messages4 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "demo.jpeg" 
            },
            {"type": "text", "text": "Describe the image."},
        ],
    }
]
messages = [messages1] * bs

texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
inputs = inputs.to("hpu")
inputs.to(torch.bfloat16)
generate_kwargs = {
    "lazy_mode": True,
    "hpu_graphs": True,
    "static_shapes": True,
    "use_cache": True,
    "cache_implementation": "static",
    "use_flash_attention": True
}
# Generate
# generate_ids = model.generate(**inputs, max_new_tokens=128, **generate_kwargs)
# output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print (output)
start = time.perf_counter()
generated_ids = model.generate(**inputs, max_new_tokens=128, **generate_kwargs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

end = time.perf_counter()
duration = end - start
total_len = 0
for res in generated_ids_trimmed:
    total_len += len(res) 
    
print(output_texts)
print("----Throughput is: ", total_len/duration, "tokens/sec")
