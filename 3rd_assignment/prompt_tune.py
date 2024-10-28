import os
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
import warnings
import argparse
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

class Config:
    def __init__(self, is_test_run: bool = False):
        self.scratch_dir = "/scratch/hmnshpl/anlp_data"
        self.cache_dir = os.path.join(self.scratch_dir, "tokenized_data")
        self.batch_size = 8
        self.num_soft_tokens = 10
        self.soft_prompt_dim = 768  # GPT-2 small's hidden size
        self.epochs = 3 if is_test_run else 10
        self.lr = 5e-5
        # Use smaller dataset for testing
        self.train_size = 1000 if is_test_run else None
        self.val_size = 100 if is_test_run else None
        self.max_input_length = 512
        self.max_target_length = 150

class SummarizationDataset:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = self._initialize_tokenizer()
        self.dataset = self._load_dataset()
        
    def _initialize_tokenizer(self) -> GPT2Tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    def _load_dataset(self):
        print('Loading Dataset...', end='\r')
        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=self.config.scratch_dir)
        
        # len(dataset["train"]), len(dataset["validation"])
        
        if self.config.train_size:
            dataset["train"] = dataset["train"].select(range(self.config.train_size))
        if self.config.val_size:
            dataset["validation"] = dataset["validation"].select(range(self.config.val_size))
            
        print('Dataset Loaded Successfully')
        return dataset
    
    def preprocess_function(self, examples: Dict) -> Dict:
        inputs = examples['article']
        targets = examples['highlights']

        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            targets,
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids

        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
        model_inputs["labels"] = labels
        return model_inputs
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # Load or create tokenized datasets
        if os.path.exists(os.path.join(self.config.cache_dir, "tokenized_dataset")):
            tokenized_datasets = load_from_disk(os.path.join(self.config.cache_dir, "tokenized_dataset"))
            print("Loaded tokenized dataset from cache.")
        else:
            tokenized_datasets = self.dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=["article", "highlights", "id"]
            )
            os.makedirs(self.config.cache_dir, exist_ok=True)
            tokenized_datasets.save_to_disk(os.path.join(self.config.cache_dir, "tokenized_dataset"))
            print("Tokenized dataset saved to cache.")

        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn
        )
        
        val_dataloader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn
        )
        
        return train_dataloader, val_dataloader
    
    def collate_fn(self, batch: list) -> Dict:
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "labels": labels}

class PromptTuningModel:
    def __init__(self, config: Config, tokenizer: GPT2Tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        
        self.model = self._initialize_model()
        self.soft_prompts = self._initialize_soft_prompts()
        self.optimizer = torch.optim.Adam([self.soft_prompts], lr=config.lr)
        
    def _initialize_model(self) -> GPT2LMHeadModel:
        model = GPT2LMHeadModel.from_pretrained("gpt2", device_map='auto')
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def _initialize_soft_prompts(self) -> torch.Tensor:
        return torch.randn(
            (self.config.num_soft_tokens, self.config.soft_prompt_dim),
            requires_grad=True,
            device=self.device
        )
    
    def prepend_prompts(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        soft_prompt_batch = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([soft_prompt_batch, self.model.transformer.wte(input_ids)], dim=1)
    
    def forward_with_prompts(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        inputs_embeds = self.prepend_prompts(input_ids)
        
        prompt_attention = torch.ones(
            (input_ids.size(0), inputs_embeds.size(1) - input_ids.size(1)),
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
        
        if labels is not None:
            pad_length = inputs_embeds.size(1) - labels.size(1)
            labels = torch.nn.functional.pad(labels, (0, pad_length), value=-100)
        
        inputs_embeds = inputs_embeds.to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)
        if labels is not None:
            labels = labels.to(torch.long)
        
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def train_epoch(self, train_dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.forward_with_prompts(input_ids, labels=labels)
            loss = outputs.loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if idx % 10 == 0:
                print(f"Batch {idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
                
        return total_loss / len(train_dataloader)

    def validate(self, val_dataloader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.eval()
        rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        metrics = {"loss": 0, "rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.forward_with_prompts(input_ids, labels=labels)
                metrics["loss"] += outputs.loss.item()
                
                predictions = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    num_beams=2,
                    no_repeat_ngram_size=2,
                    attention_mask=(input_ids != self.tokenizer.pad_token_id).long()
                )
                
                pred_str = [self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
                label_str = [self.tokenizer.decode(l[l != -100], skip_special_tokens=True) for l in labels]
                
                for pred, ref in zip(pred_str, label_str):
                    scores = rouge_scorer_fn.score(ref, pred)
                    metrics["rouge1"] += scores['rouge1'].fmeasure
                    metrics["rouge2"] += scores['rouge2'].fmeasure
                    metrics["rougeL"] += scores['rougeL'].fmeasure
        
        num_batches = len(val_dataloader)
        return tuple(value / num_batches for value in metrics.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with small dataset for testing")
    args = parser.parse_args()
    
    config = Config(is_test_run=args.test)
    dataset = SummarizationDataset(config)
    train_dataloader, val_dataloader = dataset.get_dataloaders()
    
    model = PromptTuningModel(config, dataset.tokenizer)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        avg_train_loss = model.train_epoch(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        val_loss, rouge1, rouge2, rougeL = model.validate(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"ROUGE Scores - R1: {rouge1:.4f}, R2: {rouge2:.4f}, RL: {rougeL:.4f}")

if __name__ == "__main__":
    main()