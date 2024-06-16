import torch
from transformers import BertTokenizer, BertForSequenceClassification

def analyze_sentiment(text):
    # 尝试从本地加载模型
    model_path = '../model/'
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print("本地加载模型失败，尝试从网上下载...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
        # 保存下载的模型到指定路径
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()

    if sentiment == 0:
        return "Negative"
    else:
        return "Positive"

def main():
    input_text = "待分析的文本内容"  # 这里替换为你需要分析的文本
    sentiment = analyze_sentiment(input_text)
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
