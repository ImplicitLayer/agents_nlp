import torch
from torch import nn
from transformers import GPT2Tokenizer
from agents_torch.dialogue_agent import DialogueAgent


class CustomDialogueModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=1):
        super(CustomDialogueModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, labels=None):
        # Embed inputs
        embedded = self.embedding(input_ids)
        encoder_outputs, hidden = self.encoder(embedded)

        # Use inputs as decoder inputs (teacher forcing style)
        decoder_inputs = self.embedding(input_ids)
        decoder_outputs, _ = self.decoder(decoder_inputs, hidden)

        logits = self.fc_out(decoder_outputs)

        loss = None
        if labels is not None:
            # Shift logits and labels for loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)

        return type("Output", (object,), {"logits": logits, "loss": loss})()

    def generate(self, input_ids, max_length=64, eos_token_id=None, pad_token_id=None, **kwargs):
        self.eval()
        generated = input_ids
        hidden = None

        for _ in range(max_length):
            embedded = self.embedding(generated[:, -1:])
            output, hidden = self.decoder(embedded, hidden)
            logits = self.fc_out(output[:, -1])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        return generated


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

vocab_size = tokenizer.vocab_size
model = CustomDialogueModel(vocab_size=vocab_size)

agent = DialogueAgent(model=model, tokenizer=tokenizer)

prompt = "Hello, how are you?"
print(agent.act(prompt))
