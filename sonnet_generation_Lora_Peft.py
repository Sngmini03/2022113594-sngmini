'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
from datasets import SonnetsDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
from evaluation import test_sonnet

class SonnetDatasetForHF(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.sonnets = SonnetsDataset(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.sonnets)
    def __getitem__(self, idx):
        _, text = self.sonnets[idx]
        enc = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc['labels'] = enc['input_ids'].clone()
        return enc

def train_lora(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(base_model, lora_config).to(device)
    train_dataset = SonnetDatasetForHF(args.sonnet_path, tokenizer, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir="./lora_gpt2",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,  # 커스텀 Dataset 호환을 위해 추가
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

def generate_sonnets_lora(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    model = AutoModelForCausalLM.from_pretrained(args.save_dir)

    # PEFT를 위해 LoraConfig 사용 (훈련할 때 사용한 것과 동일하게 설정)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    ))
    model.to(device)
    model.eval()

    held_out = SonnetsDataset(args.held_out_sonnet_path)

    with open(args.sonnet_out, "w", encoding="utf-8") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for idx, text in held_out:
            input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to(device)
            gen_ids = model.generate(
                input_ids,
                max_length=args.max_length,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            f.write(f"\n{idx}\n{gen_text}\n\n")
            print(f"{gen_text}\n\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
    parser.add_argument("--save_dir", type=str, default="lora_gpt2")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")  # 추가됨
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    train_lora(args)
    generate_sonnets_lora(args)
    # 평가
    score = test_sonnet(test_path=args.sonnet_out, gold_path="data/TRUE_sonnets_held_out_dev.txt")
    print(f"CHRF score: {score:.4f}")