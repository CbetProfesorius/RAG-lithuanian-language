from transformers import MarianMTModel, MarianTokenizer
from langchain.document_loaders import PDFMinerLoader, CSVLoader
import pandas as pd


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using LangChain's PDFMinerLoader."""
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    return " ".join(doc.page_content for doc in documents)

def translate_text(text, model, tokenizer, max_length=512):
    """Translates a large text in chunks."""
    sentences = text.split("\n")
    translated_texts = []
    for sentence in sentences:
        if sentence.strip():
            inputs = tokenizer(sentence, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
            translated = model.generate(**inputs)
            translated_texts.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    return " ".join(translated_texts)

def translate_csv_columns(csv_path, output_path, columns, model, tokenizer, max_length=512):
    """Translates specified columns of a CSV file."""
    df = pd.read_csv(csv_path)
    for column in columns:
        df[column] = df[column].apply(lambda x: translate_text(x, model, tokenizer, max_length) if pd.notnull(x) else x)
    df.to_csv(output_path, index=False)


model_name = "scoris/scoris-mt-en-lt"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# pdf_path = "/mnt/c/Users/askaisg/Documents/RAG-LLM/riverreport2022.pdf"
# pdf_text = extract_text_from_pdf(pdf_path)
# translated_text = translate_text(pdf_text, model, tokenizer)

# csv_path = "/mnt/c/Users/askaisg/Documents/RAG-LLM/Datasets/csv/NikeProductDescriptions_en.csv"
# output_csv_path = "/mnt/c/Users/askaisg/Documents/RAG-LLM/Datasets/csv/NikeProductDescriptions_lt.csv"
# columns_to_translate = ["Subtitle", "Product Description"]
# translate_csv_columns(csv_path, output_csv_path, columns_to_translate, model, tokenizer)
