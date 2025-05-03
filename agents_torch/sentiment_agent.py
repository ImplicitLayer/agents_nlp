import torch
from torch import nn
from transformers import PreTrainedTokenizer, AutoTokenizer
from agents_torch.base_agent import BaseAgent


class SentimentAnalysisAgent(BaseAgent):
    """
    Agent for sentiment analysis using a PyTorch model.
    """

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer = None, device: str = None):
        """
        Initializes the sentiment analysis agent with a PyTorch model and tokenizer.

        :param model: PyTorch model for sentiment analysis.
        :param tokenizer: HuggingFace tokenizer corresponding to the model.
        :param device: Device to run the model on.
        """
        super().__init__(model=model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("bert-base-uncased")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)

    def act(self, observation: str) -> int:
        """
        Predicts sentiment for the input text.

        :param observation: Input text (e.g., a review or sentence).
        :return: Predicted sentiment (0 for negative, 1 for positive).
        """
        self.model.eval()

        inputs = self.tokenizer(observation, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)

        return prediction.item()  # 0 for negative, 1 for positive

    def train_step(self, observations, actions):
        """
        Fine-tunes the model using the given sentiment analysis data.

        :param observations: List of input texts (e.g., reviews).
        :param actions: List of sentiment labels (0 or 1).
        :return: Training loss value.
        """
        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        for observation, label in zip(observations, actions):
            inputs = self.tokenizer(observation, return_tensors="pt", truncation=True, padding=True).to(self.device)
            labels = torch.tensor([label], dtype=torch.long).to(self.device)

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
        self.model = self.model.from_pretrained(path)
        self.tokenizer = self.tokenizer.from_pretrained(path)
