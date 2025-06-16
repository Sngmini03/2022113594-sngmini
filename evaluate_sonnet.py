from evaluation import test_sonnet

# 파일 경로는 필요에 따라 수정하세요.
generated_path = "predictions/generated_sonnets.txt"
gold_path = "data/TRUE_sonnets_held_out.txt"

score = test_sonnet(test_path=generated_path, gold_path=gold_path)
print(f"CHRF score: {score:.4f}")
