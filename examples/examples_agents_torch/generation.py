import torch
from torch import nn
from transformers import GPT2Tokenizer
from agents_torch.generation_agent import TextGenerationAgent


class CustomGRUTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        embedded = self.embedding(input_ids)
        outputs, _ = self.rnn(embedded)
        logits = self.fc(outputs)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type("Output", (object,), {"logits": logits, "loss": loss})()

    def generate(self, input_ids, max_length=50, eos_token_id=None, pad_token_id=None, **kwargs):
        self.eval()
        generated = input_ids
        hidden = None

        for _ in range(max_length):
            embedded = self.embedding(generated[:, -1:])
            output, hidden = self.rnn(embedded, hidden)
            logits = self.fc(output[:, -1])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        return generated


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

vocab_size = tokenizer.vocab_size

model = CustomGRUTextGenerator(vocab_size=vocab_size)
agent = TextGenerationAgent(model=model, tokenizer=tokenizer)

prompts = ["The sun is shining", "Once upon a time", "In a galaxy far"]
targets = [" and the sky is blue.", " there was a princess.", " away, a battle began."]

for epoch in range(5):
    loss = agent.train_step(prompts, targets)
    print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

prompt = "The world is changing"
generated = agent.act(prompt)
print("Generated:", generated)

