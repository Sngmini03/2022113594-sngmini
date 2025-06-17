### from google.colab import drive
###drive.mount('/content/drive')

### !pip install transformers scikit-learn optuna

# 하이브리드 모델 감성 분석
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    logging as hf_logging,
)
from torch.optim import AdamW
import torch.nn.functional as F
import optuna
from tqdm.auto import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# 1. Label Sanitiser 0/1 변환

def clean_labels(raw):
    clean = []
    for v in raw:
        if isinstance(v, str):
            v = v.strip().lower()
            clean.append(1 if v in {"positive", "pos", "1", "true"} else 0)
        else:
            clean.append(1 if int(v) > 0 else 0)
    return clean


# 2. Dataset
class PlainTextDataset(Dataset):
    def __init__(self, txt, lab):
        self.txt, self.lab = txt, lab
    def __len__(self):
        return len(self.txt)
    def __getitem__(self, idx):
        return self.txt[idx], self.lab[idx]


# 3. Load IMDB CSV

def load_imdb(path):
    tr = pd.read_csv(os.path.join(path, "imdb_train.csv"))
    te = pd.read_csv(os.path.join(path, "imdb_test.csv"))
    return (
        tr.review.tolist(),
        clean_labels(tr.sentiment),
        te.review.tolist(),
        clean_labels(te.sentiment),
    )


# 4. Model / Tokenizer Factory  (RoBERTa + XLNet)

def create_models():
    return {
        "roberta": {
            "model": RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            ),
            "tok": RobertaTokenizer.from_pretrained("roberta-base"),
        },
        "xlnet": {
            "model": XLNetForSequenceClassification.from_pretrained(
                "xlnet-base-cased", num_labels=2
            ),
            "tok": XLNetTokenizer.from_pretrained("xlnet-base-cased"),
        },
    }


# 5. Encode helper

def encode(tok, texts, device, max_len=256):
    if isinstance(texts, str):
        texts = [texts]
    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


# 6. Single-model training loop

def train_model(name, comp, X, y, device, epochs=5, batch_size=16):
    dl = DataLoader(PlainTextDataset(X, y), batch_size=batch_size, shuffle=True)
    opt = AdamW(comp["model"].parameters(), lr=2e-5)
    comp["model"].to(device).train()
    for ep in range(epochs):
        total = 0
        for txt, lab in tqdm(dl, desc=f"{name.upper()} E{ep+1}", leave=False):
            ids, att = encode(comp["tok"], txt, device)
            lab_t = torch.LongTensor(lab).to(device)
            opt.zero_grad()
            if name == "xlnet":
                out = comp["model"](
                    ids, att, token_type_ids=torch.zeros_like(ids), labels=lab_t
                )
            else:  # RoBERTa
                out = comp["model"](ids, att, labels=lab_t)
            loss = out.loss
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"{name.upper()} Epoch {ep+1}: {total/len(dl):.4f}")


# 7. Ensemble evaluation

@torch.no_grad()
def eval_ensemble(mods, weights, X, y, device, batch_size=32):
    for m in mods.values():
        m["model"].eval()
    dl = DataLoader(PlainTextDataset(X, y), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, preds = 0, []
    for txt, lab in dl:
        lab_t = torch.LongTensor(lab).to(device)
        prob = torch.zeros((len(txt), 2), device=device)
        for n, comp in mods.items():
            ids, att = encode(comp["tok"], txt, device)
            if n == "xlnet":
                logits = comp["model"](
                    ids, att, token_type_ids=torch.zeros_like(ids)
                ).logits
            else:
                logits = comp["model"](ids, att).logits
            prob += weights[n] * F.softmax(logits, dim=-1)
        tot_loss += loss_fn(prob, lab_t).item()
        preds.extend(torch.argmax(prob, 1).cpu().tolist())
    acc = accuracy_score(y, preds)
    return tot_loss / len(dl), acc, preds


# 8. Optuna objective (단일 파라미터: roberta_w)

def objective(trial, mods, Xv, yv, device):
    wr = trial.suggest_float("roberta_w", 0.0, 1.0)
    weights = {"roberta": wr, "xlnet": 1.0 - wr}
    loss, _, _ = eval_ensemble(mods, weights, Xv, yv, device)
    return loss


# 9. Main pipeline

def run(data_path, epochs=5, batch_size=16, n_trials=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 9.1 데이터 로드
    X_train, y_train, X_test, y_test = load_imdb(data_path)

    # 9.2 모델 생성
    models = create_models()

    # 9.3 개별 모델 학습
    for name, comp in models.items():
        print(f"\nTraining {name.upper()}")
        train_model(
            name,
            comp,
            X_train,
            y_train,
            device,
            epochs=epochs,
            batch_size=batch_size,
        )

    # 9.4 Optuna로 최적 가중치 탐색
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, models, X_test, y_test, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    wr = study.best_params["roberta_w"]
    weights = {"roberta": wr, "xlnet": 1.0 - wr}
    print("\nBest Weights:", weights)

    # 9.5 최종 평가
    train_loss, train_acc, _ = eval_ensemble(
        models, weights, X_train[:5000], y_train[:5000], device, batch_size
    )
    test_loss, test_acc, test_preds = eval_ensemble(
        models, weights, X_test, y_test, device, batch_size
    )
    print(
        f"\nFinal Train (5k)  -> Loss {train_loss:.4f} | Acc {train_acc:.4f}"
    )
    print(f"Final Test        -> Loss {test_loss:.4f} | Acc {test_acc:.4f}")

    # 9.6 분류 보고서
    print(
        "\nClassification Report:\n",
        classification_report(y_test, test_preds, target_names=["negative", "positive"]),
    )

# 10. Entry point

if __name__ == "__main__":
    run(
        "/content/drive/MyDrive/nlp_final/aclImdb/data",
        epochs=3,        # ↔ 필요에 따라 조정
        batch_size=8,
        n_trials=10,
    )
