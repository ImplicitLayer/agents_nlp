from transformers import AutoModelForCausalLM, AutoTokenizer
from agents_transformers.generation_agent import TextGenerationAgent

model_name = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

agent = TextGenerationAgent(model=model, tokenizer=tokenizer, max_length=50)

prompt = "Once upon a time in a distant galaxy,"
generated_text = agent.act(prompt)
print("Generated:", generated_text)

observations = ["The capital of France is"]
targets = ["Paris."]
loss = agent.train_step(observations, targets)
print("Training loss:", loss)
