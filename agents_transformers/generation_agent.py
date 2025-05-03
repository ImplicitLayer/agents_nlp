from typing import Any
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
import torch
from agents_transformers.base_agent import BaseAgent


class TextGenerationAgent(BaseAgent):
    """
    Agent for text generation tasks using causal language models (e.g., GPT2, LLaMA).
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer = None, device: str = None,
                 max_length: int = 50):
        """
        Initializes the agent with a language model and tokenizer.

        :param model: A pretrained causal language model (e.g., GPT2).
        :param tokenizer: Corresponding tokenizer for the model.
        :param device: Device to run the model on ("cpu" or "cuda"). Auto-detected if None.
        :param max_length: Maximum length of generated responses.
        """
        super().__init__(model=model)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)

    def act(self, observation: Any) -> Any:
        """
        Generates a text response given the input prompt.

        :param observation: Input prompt string.
        :return: Generated text response.
        """
        if not isinstance(observation, str):
            raise ValueError("Input observation must be a string.")

        inputs = self.tokenizer(observation, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_length=self.max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def train_step(self, observations: Any, actions: Any):
        """
        Fine-tunes the model using teacher forcing on given prompts and expected outputs.

        :param observations: List of input prompts.
        :param actions: List of expected generated texts.
        """
        if not isinstance(observations, list) or not isinstance(actions, list):
            raise ValueError("Observations and actions must be lists of strings.")
        if len(observations) != len(actions):
            raise ValueError("Observations and actions must have the same length.")

        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        for prompt, expected in zip(observations, actions):
            # Объединяем prompt и target для обучения автопорождения
            input_text = prompt + self.tokenizer.eos_token + expected
            encodings = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            labels = encodings["input_ids"].clone()
            outputs = self.model(**encodings, labels=labels)
            loss = outputs.loss
            loss.backward()
            losses.append(loss.item())

        self.optimizer.step()
        self.model.eval()
        return sum(losses) / len(losses)

    def save(self, path: str):
        """
        Saves the model and tokenizer to the specified path.

        :param path: Destination directory.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """
        Loads the model and tokenizer from the specified path.

        :param path: Directory containing model files.
        """
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
