from transformers import MarianMTModel, MarianTokenizer

# ✅ Correct model for English → Tamil
model_name = "Helsinki-NLP/opus-mt-en-ta"


# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Read English lines from train.en
print("Reading train.en...")
with open("data/train.en", "r", encoding="utf-8") as f:
    en_lines = [line.strip() for line in f.readlines() if line.strip()]

# Optional: use only first 10 lines for testing (remove this later)
# en_lines = en_lines[:10]

# Translate in batches
ta_lines = []
batch_size = 8

print("Translating to Tamil...")
for i in range(0, len(en_lines), batch_size):
    batch = en_lines[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ta_lines.extend(translations)
    print(f"Translated {i + len(batch)} / {len(en_lines)}")

# Save the translated lines to train.ta
print("Saving to train.ta...")
with open("data/train.ta", "w", encoding="utf-8") as f:
    for line in ta_lines:
        f.write(line + "\n")

print("✅ Done! `train.ta` generated successfully.")
