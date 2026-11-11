
from pathlib import Path
import re

# Step 1: Locate the dataset
data_path = Path("data/dataset.txt")

# Step 2: Read the raw text
with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()

print("ORIGINAL DATASET:")
print(text[:300])

# Step 3: Basic text cleaning
# Remove extra spaces and normalize line endings
cleaned_text = re.sub(r"\s+", " ", text).strip()

# Step 4: Dataset statistics
words = cleaned_text.split()

print("\nDATASET STATISTICS")
print("Total characters:", len(cleaned_text))
print("Total words:", len(words))
print("Unique words:", len(set(words)))

# Step 5: Save cleaned text
output_path = Path("data/cleaned_dataset.txt")

with open(output_path, "w", encoding="utf-8") as file:
    file.write(cleaned_text)

print("\nCleaned dataset saved successfully!")
