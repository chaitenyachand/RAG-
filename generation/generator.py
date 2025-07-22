from transformers import pipeline

def generate_answer(contexts, question):
    combined_context = " ".join(contexts)
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = f"Context: {combined_context}\nQuestion: {question}\nAnswer:"
    output = generator(prompt, max_length=100, do_sample=False)
    return output[0]["generated_text"]
