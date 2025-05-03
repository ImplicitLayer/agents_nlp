import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from agents_transformers.base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    """
    A sentiment analysis agent that classifies input text into sentiment categories.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, lr: float = 2e-5):
        """
        Initialize the sentiment analysis agent.

        :param model: Hugging Face classification model (e.g., BERT, RoBERTa).
        :param tokenizer: Corresponding tokenizer.
        :param lr: Learning rate for optimizer.
        """
        super().__init__(model=model)
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer
        self.device = self.model.device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()

    def act(self, observation):
        """
        Predict sentiment class for the given input text.

        :param observation: Input text string.
        :return: Predicted class index (int).
        """
        if not isinstance(observation, str):
            raise ValueError("Observation must be a string.")

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(observation, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        self.model.train()
        return prediction

    def train_step(self, observations, actions):
        """
        Perform a training step on input-output pairs.

        :param observations: List of input texts.
        :param actions: List of expected sentiment class indices.
        :return: Training loss (float).
        """
        if not (isinstance(observations, list) and isinstance(actions, list) and len(observations) == len(actions)):
            raise ValueError("Observations and actions must be lists of equal length.")

        inputs = self.tokenizer(observations, return_tensors="pt", padding=True, truncation=True).to(self.device)
        labels = torch.tensor(actions).to(self.device)

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        """
        Save model and tokenizer to disk.

        :param path: Directory path to save the model.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """
        Load model and tokenizer from disk.

        :param path: Directory path to load the model from.
        """
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
