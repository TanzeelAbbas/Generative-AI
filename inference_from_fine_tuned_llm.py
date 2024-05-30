# load fine tuned model and generate response

import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel, get_peft_model, LoraConfig

class ModelLoader:
    def __init__(self):
        self.pipe = None

    def load_model(self, base_model_name, fine_tuned_model):

        os.environ["HF_TOKEN"] = "PASTE YOUR HUGGING FACE API HERE"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            token=os.environ["HF_TOKEN"],
        )
        # peft_config = LoraConfig(
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     r=64,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )

        # model2= get_peft_model(base_model, peft_config)
        
        #this function returns the outputs from the model received, and inputs.
        # def get_outputs(model, inputs, max_new_tokens=100):
        #     outputs = model.generate(
        #         input_ids=inputs["input_ids"],
        #         attention_mask=inputs["attention_mask"],
        #         max_new_tokens=max_new_tokens,
        #         repetition_penalty=1.5, #Avoid repetition.
        #         early_stopping=True, #The model can stop before reach the max_length
        #         eos_token_id=tokenizer.eos_token_id
        #     )
        #     return outputs
            

        model = PeftModel.from_pretrained(base_model, fine_tuned_model)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # input_sentences = tokenizer("What could be the reason behind a tablet battery showing 0 percent charging percentage?", return_tensors="pt") 
        # foundational_outputs_sentence = get_outputs(model, input_sentences, max_new_tokens=1024)

        # print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.pipe = pipeline(
            "text-generation", model, tokenizer=tokenizer, max_length=1024
        )

    def generate_response(self, prompt):
        # result = self.pipe(f"{prompt}")
        result = self.pipe(f"{prompt}")
        return result[0]['generated_text']
    
    
if __name__ == "__main__":
    model_loader = ModelLoader()
    base_model ="microsoft/Phi-3-mini-4k-instruct" 
    fine_tuned_model = "phi-3-mini-fine-tuned"  
    pipe = model_loader.load_model(base_model, fine_tuned_model)
    prompt = "What is forecast engine?"  
    response = model_loader.generate_response(prompt)
    print(response)
