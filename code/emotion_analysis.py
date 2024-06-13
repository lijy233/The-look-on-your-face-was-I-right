import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

def analyze_sentiment(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()

    if sentiment == 0:
        return "Negative"
    else:
        return "Positive"
