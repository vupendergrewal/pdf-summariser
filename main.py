import fitz
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def summarise_text(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary)

def save_summary(summary, output_path="summary.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to {output_path}")

# --- MAIN PROGRAM ---
pdf_path = input("Enter PDF file path: ").strip().strip('"')

print("\nExtracting text...")
text = extract_text_from_pdf(pdf_path)

if not text.strip():
    print("No text found. PDF might be scanned.")
else:
    print("Summarising...")
    summary = summarise_text(text, num_sentences=7)
    print("\n--- SUMMARY ---\n")
    print(summary)
    save_summary(summary)