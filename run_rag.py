import gradio as gr
import torch
import os
import pickle
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, MarianMTModel, MarianTokenizer
from langchain.document_loaders import PDFMinerLoader, CSVLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.text_splitter import SentenceTransformersTokenTextSplitter


template = """
{context}

{question}
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

MODEL_NAME = "TheBloke/Llama-2-13B-chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            # model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

TRANSLATION_MODEL_NAME_EN = "scoris/scoris-mt-lt-en"
translation_tokenizer_en = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME_EN)
translation_model_en = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME_EN)

TRANSLATION_MODEL_NAME_LT = "scoris/scoris-mt-en-lt"
translation_tokenizer_lt = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME_LT)
translation_model_lt = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME_LT)

current_file_hash, current_db, current_embeddings = None, None, None

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)


def translate_to_english(text):
    """
    Translates text from Lithuanian to English using the Marian MT model.
    """
    inputs = translation_tokenizer_en([text], return_tensors="pt", padding=True)
    outputs = translation_model_en.generate(**inputs)
    return translation_tokenizer_en.decode(outputs[0], skip_special_tokens=True)

def translate_to_lithuanian(text):
    """
    Translates text from Lithuanian to English using the Marian MT model.
    """
    lines = text.split('\n')
    translated_lines = []

    for line in lines:
        if line.strip():
            inputs = translation_tokenizer_lt([line], return_tensors="pt", padding=True)
            outputs = translation_model_lt.generate(**inputs)
            translated_line = translation_tokenizer_lt.decode(outputs[0], skip_special_tokens=True)
            translated_lines.append(translated_line)
        else:
            translated_lines.append("")
    return '\n'.join(translated_lines)

def save_cache(filename, data):
    with open(os.path.join(CACHE_DIR, filename), 'wb') as f:
        pickle.dump(data, f)

def load_cache(filename):
    try:
        with open(os.path.join(CACHE_DIR, filename), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    
def process_files(file_paths):
    try:
        cache_key = f'{hash(tuple(file_paths))}.pkl'
        cached_data = load_cache(cache_key)

        if cached_data:
            print("Cache hit!")
            return cached_data
        
        print("Processing files...")
        docs = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                loader = PDFMinerLoader(file_path)
                docs.extend(loader.load())
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
                docs.extend(loader.load())
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs.extend(loader.load())

        print("Files loaded.")
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=64)
        texts = text_splitter.split_documents(docs)
        print("Text split into chunks.")

        db = FAISS.from_documents(texts, embeddings)
        print("FAISS index created.")
        save_cache(cache_key, (db, embeddings))
        return db, embeddings
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def query_files(files, question):
    if not files or not question.strip():
        return "Please upload valid files and enter a question."

    print("Starting query processing...")

    english_question = translate_to_english(question)

    db, embeddings = process_files(files)
    
    if db is None:
        print("Error during processing.")
        return f"Error processing files: {embeddings}"

    print("Processing complete.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain(english_question)

    output_text_en = result["result"].split(english_question)[-1].strip()
    output_text_lt = translate_to_lithuanian(output_text_en)

    return output_text_lt



with gr.Blocks() as interface:
    gr.Markdown("### RAG užduotis")
    gr.Markdown(
        "Įkelkite dokumentų failus (PDF, CSV, DOCX) ir užduokite klausimą.")

    with gr.Row():
        question_input = gr.Textbox(label="Užduokite klausimą", lines=3)
        files_input = gr.File(label="Įkelkite failus", type="filepath", file_count="multiple")

    submit_button = gr.Button("Patvirtinti")
    output_text = gr.Textbox(label="Atsakymas", lines=10)
    submit_button.click(query_files, inputs=[files_input, question_input], outputs=output_text)

if __name__ == "__main__":
    interface.launch()