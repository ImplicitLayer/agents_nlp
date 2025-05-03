"""
==============================================
GPT2
==============================================
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from agents_transformers.dialogue_agent import DialogueAgent

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

agent = DialogueAgent(model=model, tokenizer=tokenizer, max_length=50)

prompt = "What is the capital of France?"
response = agent.act(prompt)
print("Generated response:", response)

observations = [
    "What is the capital of France?",
    "Who wrote '1984'?",
    "What is the boiling point of water?"
]
actions = [
    "The capital of France is Paris.",
    "George Orwell wrote '1984'.",
    "The boiling point of water is 100 degrees Celsius."
]

loss = agent.train_step(observations, actions)
print(f"Training loss: {loss:.4f}")

agent.save("dialogue_agent_gpt2")

"""
==============================================
EleutherAI
==============================================
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from agents_transformers.dialogue_agent import DialogueAgent


# 1. Загрузка модели и токенизатора
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Установка pad_token, так как GPT-Neo его не имеет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# 3. Создание агента
agent = DialogueAgent(model=model, tokenizer=tokenizer, max_length=50)

# 4. Пример генерации
prompt = "What is the capital of France?"
response = agent.act(prompt)
print("Generated:", response)

# 5. Пример обучения
observations = ["What is the capital of France?", "Who wrote 1984?"]
actions = ["Paris is the capital of France.", "George Orwell wrote 1984."]
loss = agent.train_step(observations, actions)
print("Training loss:", loss)

"""
==============================================
T5
==============================================
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from agents_transformers.dialogue_agent import DialogueAgent

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

agent = DialogueAgent(model=model, tokenizer=tokenizer, max_length=50)


user_input = "How are you today?"

response = agent.act(user_input)
print(f"Agent: {response}")

observations = ["Hello, how are you?", "What is your name?"]
actions = ["I'm good, thank you!", "I'm a T5-based chatbot."]

loss = agent.train_step(observations, actions)
print(f"Training loss: {loss}")

