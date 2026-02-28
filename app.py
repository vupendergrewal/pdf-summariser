from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import fitz
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def summarise_text(text, num_sentences=7):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarise", methods=["POST"])
def summarise():
    file = request.files['pdf']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text(filepath)

    if not text.strip():
        return render_template("result.html", error="No text found. PDF might be scanned.")

    summary = summarise_text(text)
    word_count = len(text.split())
    sentence_count = summary.count('.') + 1

    return render_template("result.html",
        summary=summary,
        filename=filename,
        word_count=word_count,
        sentence_count=sentence_count,
        error=None
    )

if __name__ == "__main__":
    app.run()