import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class ABSAAspectModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ABSAAspectModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = self.loss_fn(outputs.logits, labels)
        return outputs.logits, loss

class ABSASentimentModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ABSASentimentModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = self.loss_fn(outputs.logits, labels)
        return outputs.logits, loss
