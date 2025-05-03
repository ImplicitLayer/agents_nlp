from torch import nn
from transformers import BertModel
from transformers import AutoTokenizer
from agents_torch.sentiment_agent import SentimentAnalysisAgent


class CustomSentimentModel(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=2):
        super(CustomSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return type("Output", (object,), {"logits": logits, "loss": loss})()


model = CustomSentimentModel(pretrained_model_name="bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

agent = SentimentAnalysisAgent(model=model, tokenizer=tokenizer)

text = "I really love this product!"
print("Predicted sentiment:", agent.act(text))

observations = ["I hate this", "It is awesome", "Terrible experience", "Really good one"]
labels = [0, 1, 0, 1]

loss = agent.train_step(observations, labels)
print("Training loss:", loss)
