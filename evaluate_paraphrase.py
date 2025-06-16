import torch
from models.gpt2 import GPT2Model
from paraphrase_detection import ParaphraseGPT, add_arguments
from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import model_eval_paraphrase
from torch.utils.data import DataLoader

# 사용자 환경에 맞게 수정하세요
class Args:
    para_dev = "data/quora-dev.csv"
    model_size = "gpt2"
    d = 768
    l = 12
    num_heads = 12
    batch_size = 8
    use_gpu = True
    filepath = "10-1e-05-paraphrase.pt"  # 저장된 모델 파일명

args = Args()
args = add_arguments(args)
device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

# 데이터 준비
dev_data = load_paraphrase_data(args.para_dev)
dev_dataset = ParaphraseDetectionDataset(dev_data, args)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

# 모델 로드
saved = torch.load(args.filepath)
model = ParaphraseGPT(saved['args'])
model.load_state_dict(saved['model'])
model = model.to(device)
model.eval()

# 평가
acc, f1, y_pred, y_true, sent_ids = model_eval_paraphrase(dev_loader, model, device)
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
