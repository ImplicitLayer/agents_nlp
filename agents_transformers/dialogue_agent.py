import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from agents_transformers.base_agent import BaseAgent


class DialogueAgent(BaseAgent):
    """
    A dialogue agent for the MultiAgentDialogueEnv.
    Generates responses using a Hugging Face transformer model.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 50, lr: float = 5e-5):
        """
        Initialize the agent with a given model and tokenizer.

        :param model: Hugging Face generative model (e.g., GPT-2, T5, etc.)
        :param tokenizer: Matching tokenizer.
        :param max_length: Max tokens to generate in response.
        :param lr: Learning rate for training.
        """
        super().__init__(model=model)
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = self.model.device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()  # Make sure model is in training mode for fine-tuning

    def act(self, observation):
        """
        Generate a response based on the input observation.

        :param observation: Input prompt string.
        :return: Generated text response.
        """
        if not isinstance(observation, str):
            raise ValueError("Observation must be a string.")

        inputs = self.tokenizer(observation, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                inputs.input_ids,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def train_step(self, observations, actions):
        """
        Perform a single fine-tuning step using input-output pairs.

        :param observations: List of input strings (e.g., questions or prompts).
        :param actions: List of expected outputs (e.g., correct responses).
        """
        if not (isinstance(observations, list) and isinstance(actions, list) and len(observations) == len(actions)):
            raise ValueError("Both observations and actions must be lists of equal length.")

        self.model.train()

        # Prepare tokenized inputs and labels
        inputs = self.tokenizer(observations, return_tensors="pt", padding=True, truncation=True).to(self.device)
        labels = self.tokenizer(actions, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        # Replace padding token id with -100 so they're ignored in the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        """
        Save the model and tokenizer to the specified directory.

        :param path: Output directory for saving.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """
        Load the model and tokenizer from a directory.

        :param path: Path to the directory.
        """
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
