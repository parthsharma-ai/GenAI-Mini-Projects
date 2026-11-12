
import re
from transformers import AutoTokenizer


class TokenizerComparison:

    def __init__(self):
        print("Loading Hugging Face tokenizers...")

        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(
            "gpt2"
        )

        print("Tokenizers loaded successfully!\n")

    # Simple word-level tokenizer
    def word_tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text)

    # BERT tokenizer
    def bert_tokenize(self, text):
        encoded = self.bert_tokenizer(
            text,
            add_special_tokens=True
        )

        tokens = self.bert_tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        return {
            "tokens": tokens,
            "token_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_count": len(encoded["input_ids"])
        }

    # GPT-2 tokenizer
    def gpt2_tokenize(self, text):
        encoded = self.gpt2_tokenizer(
            text,
            add_special_tokens=False
        )

        tokens = self.gpt2_tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        return {
            "tokens": tokens,
            "token_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_count": len(encoded["input_ids"])
        }

    # Compare all three tokenizers
    def compare(self, text):

        print("=" * 65)
        print("TOKENIZER COMPARISON")
        print("=" * 65)

        print("\nOriginal Text:")
        print(text)

        # Word-level
        word_tokens = self.word_tokenize(text)

        print("\n1. WORD-LEVEL TOKENIZER")
        print("Tokens:", word_tokens)
        print("Token Count:", len(word_tokens))

        # BERT
        bert_result = self.bert_tokenize(text)

        print("\n2. BERT TOKENIZER")
        print("Tokens:", bert_result["tokens"])
        print("Token IDs:", bert_result["token_ids"])
        print("Attention Mask:", bert_result["attention_mask"])
        print("Token Count:", bert_result["token_count"])

        # GPT-2
        gpt2_result = self.gpt2_tokenize(text)

        print("\n3. GPT-2 TOKENIZER")
        print("Tokens:", gpt2_result["tokens"])
        print("Token IDs:", gpt2_result["token_ids"])
        print("Attention Mask:", gpt2_result["attention_mask"])
        print("Token Count:", gpt2_result["token_count"])

        print("\n" + "=" * 65)

        print("\nTOKEN COUNT SUMMARY")
        print("-" * 40)

        print(f"Word-level : {len(word_tokens)}")
        print(f"BERT       : {bert_result['token_count']}")
        print(f"GPT-2      : {gpt2_result['token_count']}")

        return {
            "word_level": word_tokens,
            "bert": bert_result,
            "gpt2": gpt2_result
        }


if __name__ == "__main__":

    comparison = TokenizerComparison()

    sample_text = (
        "Generative AI is transforming the world! "
        "I am building intelligent AI applications using Python."
    )

    comparison.compare(sample_text)
