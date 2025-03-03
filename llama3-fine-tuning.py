# Fine tuning LLM with LoRA Script

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer
import pandas as pd

class FineTuning:
    def __init__(self, model_name, output_dir, lora_r, lora_alpha, lora_dropout):
        # Model name
        self.model_name = model_name
        # Output directory for training
        self.output_dir = output_dir
        # Rank of LoRA matrices
        self.lora_r = lora_r
        # Scaling factor for LoRA
        self.lora_alpha = lora_alpha
        # Dropout rate for LoRA layers
        self.lora_dropout = lora_dropout

    def load_data(self, train_path, test_path):
        train_dataset = pd.read_csv(train_path)
        test_dataset = pd.read_csv(test_path)
        # For the chat model(LLM) we convert the dataset to below format for training. 
        # Note: Below is specific format for llama chat model fine tuning for other it may be different format
        train_dataset['text'] =  train_dataset['Questions'] + train_dataset['Answers'] 
        train_dataset = Dataset.from_pandas(train_dataset[['text']])
        test_dataset['text'] =  test_dataset['Questions'] + test_dataset['Answers'] 
        test_dataset = Dataset.from_pandas(test_dataset[['text']])
        return train_dataset, test_dataset

    def setup_model(self):
        os.environ["HF_TOKEN"] = "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"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            token=os.environ["HF_TOKEN"],
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        return model

    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, token=os.environ["HF_TOKEN"],)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def setup_lora(self):
        lora_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return lora_config

    def setup_training_arguments(self):
        # Training arguments
        training_arguments = TrainingArguments(
            # Output directory for training
            output_dir=self.output_dir,
            # Overwrite output directory if it already exists
            overwrite_output_dir=True,
            # Number of epochs to train for
            num_train_epochs=1,
            # we can also use 'steps'
            evaluation_strategy="epoch", 
            # Batch size per device during training
            per_device_train_batch_size=4,
            # Number of gradient accumulation steps
            gradient_accumulation_steps=1,
            # Optimizer to use
            optim="paged_adamw_32bit",
            # Number of steps to save the model checkpoint
            save_steps=5000,
            # Number of steps to log metrics
            logging_steps=5000,
            # Initial learning rate
            learning_rate=2e-4,
            # Weight decay for regularization
            weight_decay=0.001,
            # Use mixed precision training
            fp16=False,
            # Use bfloat16 precision with A100 GPUs
            bf16=False,
            # Maximum norm for gradient clipping
            max_grad_norm=0.3,
            # Total number of training steps or -1 for number of epochs * number of training samples
            max_steps=-1,
            # Warmup ratio for learning rate scheduler
            warmup_ratio=0.03,
            # Group sequences by length for efficient batching
            group_by_length=True,
            # Learning rate scheduler type
            lr_scheduler_type="cosine",
            # Report metrics to TensorBoard
            report_to="tensorboard"
        )
        return training_arguments

    def setup_trainer(self, model, tokenizer, lora_config, train_dataset, test_dataset):
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=self.setup_training_arguments(),
            packing=False,
        )
        return trainer

    def train(self, train_dataset,test_dataset, fine_tuned_model_name):
        model = self.setup_model()
        tokenizer = self.setup_tokenizer()
        lora_config = self.setup_lora()
        trainer = self.setup_trainer(model, tokenizer, lora_config, train_dataset, test_dataset)
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        lora_model = get_peft_model(trainer.model, self.setup_lora())
        parameters = lora_model.print_trainable_parameters()
        fine_tuned_model = fine_tuned_model_name
        trainer.model.save_pretrained(fine_tuned_model)
        return trainer


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    ft = FineTuning(model_name, "results", 64, 16, 0.1) # "meta-llama/Meta-Llama-3-8B-Instruct"
    train_dataset, test_dataset = ft.load_data("jira_QA.csv", "FAQs.csv")
    fine_tuned_model_name = "Llama-3-8b-finetuned"
    ft.train(train_dataset, test_dataset, fine_tuned_model_name)
    
    

# Check the plots on tensorboard, as follows

# %load_ext tensorboard
# %tensorboard --logdir results/runs
