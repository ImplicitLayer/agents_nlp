from transformers import AutoTokenizer, AutoModelForSequenceClassification
from agents_transformers.sentiment_agent import SentimentAgent

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

agent = SentimentAgent(model=model, tokenizer=tokenizer)

text = "I really enjoyed this movie!"
sentiment = agent.act(text)
print(f"Predicted sentiment: {sentiment} (0 = negative, 1 = positive)")

observations = ["I love this!", "This is terrible."]
labels = [1, 0]

loss = agent.train_step(observations, labels)
print(f"Training loss: {loss}")
