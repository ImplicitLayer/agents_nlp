# NLP Agents

This project provides a modular architecture for building and training **agents 
for natural language processing (NLP)** tasks using both pre-trained models from 
the 🤗 `transformers` library and custom models in pure `PyTorch`.

Tasks solved by agents:

* Sentiment Analysis

* Dialogue Generation

* Text Generation

The project focuses on flexibility: you can easily switch between
standard models (e.g. `BERT` or `GPT-2`) and your own architectures 
(e.g. `GRU`, `LSTM`, `Transformer`, etc.).

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ImplicitLayer/agents_nlp.git
cd agents_nlp
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Functionality 

Each agent implements a common interface:

* `act(observation)` - accepts text input and returns the result (e.g. class label or generated text)

* `train_step(observations, actions)` - training method (used for training)

* `save(path)` / `load(path)` - save and load model and tokenizer

Agents can work:

* With any models from `transformers` (BERT, GPT, RoBERTa, etc.)

* With custom PyTorch models defined by user

## Examples of use

**Sentiment Analysis with BERT**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from agents_torch.sentiment_agent import SentimentAnalysisAgent

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

agent = SentimentAnalysisAgent(model=model, tokenizer=tokenizer)
print(agent.act("I love this movie!"))
```

**Text generation**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from agents_torch.generation_agent import TextGenerationAgent

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

agent = TextGenerationAgent(model=model, tokenizer=tokenizer)
print(agent.act("In the future, AI will"))
```

**Custom model**

```python
from models.custom_gru import CustomGRUTextGenerator
from transformers import GPT2Tokenizer
from agents_torch.generation_agent import TextGenerationAgent

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = CustomGRUTextGenerator(vocab_size=tokenizer.vocab_size)

agent = TextGenerationAgent(model=model, tokenizer=tokenizer)
print(agent.act("Once upon a time"))
```

## Training the model
Each agent implements a `train_step` method that allows it to 
pre-train the model on its data:

```python
prompts = ["Once upon a time", "In a galaxy far"]
targets = [" there was a dragon.", " away, war began."]

for epoch in range(3):
    loss = agent.train_step(prompts, targets)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
```
