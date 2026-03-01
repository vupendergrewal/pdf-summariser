from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import fitz
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os
import io
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import json

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
        return render_template(
            "result.html",
            error="No text found. PDF might be scanned.",
            summary=None,
            filename=None,
            word_count=0,
            sentence_count=0,
            document_text=None
        )

    summary = summarise_text(text)
    word_count = len(text.split())
    sentence_count = summary.count('.') + 1

    return render_template(
        "result.html",
        summary=summary,
        filename=filename,
        word_count=word_count,
        sentence_count=sentence_count,
        document_text=text[:8000],  # important
        error=None
    )

@app.route("/download/word", methods=["POST"])
def download_word():
    summary = request.form.get("summary")
    filename = request.form.get("filename")

    doc = Document()
    doc.add_heading("PDF Summary", 0)
    doc.add_paragraph(f"Original file: {filename}")
    doc.add_paragraph(" ")
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
        download_name="summary.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

@app.route("/download/pdf", methods=["POST"])
def download_pdf():
    summary = request.form.get("summary")
    filename = request.form.get("filename")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 60, "PDF Summary")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 90, f"Original file: {filename}")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 130, "Summary:")
    c.setFont("Helvetica", 11)

    y = height - 160
    words = summary.split()
    line = ""
    for word in words:
        if c.stringWidth(line + word, "Helvetica", 11) < width - 100:
            line += word + " "
        else:
            c.drawString(50, y, line)
            y -= 20
            line = word + " "
            if y < 50:
                c.showPage()
                y = height - 50
    c.drawString(50, y, line)
    c.save()

    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
        download_name="summary.pdf",
        mimetype="application/pdf")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    document_text = data.get("document_text")

    if not document_text:
        return {"answer": "Document missing. Please upload again."}

    document_text = document_text[:8000]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer only based on the provided document."},
                {"role": "user", "content": f"Document:\n{document_text}\n\nQuestion: {question}"}
            ]
        )

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}