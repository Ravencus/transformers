import time
from tqdm import tqdm

def generate_with_time(model, inputs, **kwargs):
    start_time = time.time()
    outputs = model.generate(**inputs, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time

def assisted_generate_with_time(model, assistant_model, inputs, **kwargs):
    start_time = time.time()
    # outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5, **kwargs)
    outputs = model.generate(**inputs, assistant_model=assistant_model, **kwargs)
    generation_time = time.time() - start_time
    return outputs, generation_time

if __name__ == "__main__":
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    prompt = "How to measure my generation time?"
    
    checkpoint = "EleutherAI/pythia-1.4b-deduped"
    assistant_checkpoint = "EleutherAI/pythia-160m-deduped"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
    
    raw_time = generate_with_time(model, inputs)
    assisted_time = assisted_generate_with_time(model, assistant_model, inputs)
    print("raw:")
    print(raw_time)
    print("assisted:")
    print(assisted_time)