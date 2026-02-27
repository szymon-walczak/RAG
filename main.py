import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

import whisper
import fitz  # PyMuPDF
from moviepy import VideoFileClip
import moviepy.video.fx as fx

load_dotenv()
DB_DIR = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
whisper_model = None

def get_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base", device="cpu")
    return whisper_model

def process_video_to_fast_audio(video_path):
    """Extracts audio and speeds it up 1,5x using MoviePy 2.0 syntax."""
    base_name = os.path.splitext(video_path)[0]
    output_audio = f"{base_name}_fast.mp3"
    
    with VideoFileClip(video_path) as video:
        # Speed up and extract audio
        fast_video = video.with_effects([fx.MultiplySpeed(1.5)])
        fast_video.audio.write_audiofile(output_audio, fps=44100, logger=None)
    return output_audio

def save_debug_txt(filename, text):
    """Saves the raw extracted text to a debug folder."""
    debug_folder = "debug_txt"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    # Create a txt filename based on the original file
    debug_file = os.path.join(debug_folder, f"{filename}.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   [DEBUG] Raw text saved to: {debug_file}")

def process_directory(path):
    documents = []
    for filename in os.listdir(path):
        f_path = os.path.join(path, filename)
        text = ""

        # 1. Extract Text
        if filename.endswith(".pdf"):
            print(f"Processing PDF: {filename}")
            with fitz.open(f_path) as doc:
                text = "".join([p.get_text() for p in doc])

        elif filename.endswith(".mp4"):
            print(f"Processing Video: {filename}")
            fast_audio = process_video_to_fast_audio(f_path)
            text = get_whisper().transcribe(fast_audio, fp16=False)["text"]
            #os.remove(fast_audio)

        elif filename.endswith((".mp3", ".wav")):
            print(f"Processing Audio: {filename}")
            text = get_whisper().transcribe(f_path, fp16=False)["text"]

        elif filename.endswith(".txt"):
            print(f"Processing Text File: {filename}")
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Fallback for files saved with different encoding (common in Windows)
                with open(f_path, "r", encoding="latin-1") as f:
                    text = f.read()

        # 2. DEBUG: Save raw text before chunking
        if text.strip() and not filename.endswith(".txt"):
            save_debug_txt(filename, text)

            # 3. Chunking & DB Loading
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            for c in chunks:
                documents.append(Document(page_content=c, metadata={"source": filename}))

    if documents:
        vector_db.add_documents(documents)
        print(f"\n--- SUCCESS: Added {len(documents)} chunks to DB ---")


def ask_question(query):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        thinking_budget=1024
    )
    
    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response["answer"]


def main():
    while True:
        print("\n1. Load Data | 2. Query | 3. Exit")
        choice = input("Choice: ")
        if choice == '1':
            process_directory(input("Folder path: "))
        elif choice == '2':
            print(f"\nAI: {ask_question(input('Question: '))}")
        elif choice == '3':
            break

if __name__ == "__main__":
    main()
