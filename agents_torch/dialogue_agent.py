import torch
from torch import nn
from transformers import PreTrainedTokenizer, AutoTokenizer
from agents_torch.base_agent import BaseAgent


class DialogueAgent(BaseAgent):
    """
    Agent for dialogue tasks using a PyTorch model.
    """

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer = None, device: str = None, max_length: int = 64):
        """
        Initializes the dialogue agent with a PyTorch model and tokenizer.

        :param model: PyTorch model for text generation.
        :param tokenizer: HuggingFace tokenizer corresponding to the model.
        :param device: Device to run the model on.
        :param max_length: Maximum number of tokens to generate.
        """
        super().__init__(model=model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def act(self, observation: str) -> str:
        """
        Generates a response to the input message.

        :param observation: Input message (prompt).
        :return: Generated reply.
        """
        self.model.eval()

        inputs = self.tokenizer(observation, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def train_step(self, observations, actions):
        """
        Fine-tunes the model using the given dialogue data (teacher forcing).

        :param observations: List of input prompts (e.g., user messages).
        :param actions: List of expected outputs (e.g., agent responses).
        """
        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        for prompt, target in zip(observations, actions):
            input_text = prompt + self.tokenizer.eos_token + target
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            labels = inputs["input_ids"].clone()

            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            losses.append(loss.item())

        self.optimizer.step()
        self.model.eval()
        return sum(losses) / len(losses)

    def save(self, path: str):
        """
        Saves the model and tokenizer.

        :param path: Path to the save directory.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """
        Loads the model and tokenizer.

        :param path: Path to the saved model directory.
        """
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
