RAG (Retrieval-Augmented Generation) is a powerful approach that combines retrieval techniques with generative models to enhance the quality and relevance of generated content. 

````
brew install ffmpeg poetry pyenv
pyenv install 3.12
pyenv local 3.12
poetry install
````

export GOOGLE_API_KEY=AIza....
or create .env file with GOOGLE_API_KEY=...

``
poetry run python main.py
``

it will create local chroma db, download whisper model and locally process all the videos and pdfs in the selected folder, 
then you can ask questions about the content of those files.

````
1. Load Data | 2. Query | 3. Exit
   Choice: 1
   Folder path: data
   Processing Video: 2 part Databases for GenAI.mp4
   Loading Whisper model...
   [DEBUG] Raw text saved to: debug_txt/2 part Databases for GenAI.mp4.txt
   Processing PDF: Databases for GenAI.pdf
   [DEBUG] Raw text saved to: debug_txt/Databases for GenAI.pdf.txt
   Processing Video: 1 part. RAG Intro.mp4
   [DEBUG] Raw text saved to: debug_txt/1 part. RAG Intro.mp4.txt
   Processing PDF: 1.pdf
   [DEBUG] Raw text saved to: debug_txt/1.pdf.txt


--- SUCCESS: Added 68 chunks to DB ---

1. Load Data | 2. Query | 3. Exit
   Choice: 2
   Question: What are the production 'Do's' for RAG?

AI: Based on the provided context, one production 'Do' for RAG is:

*   **Consider PostgreSQL with its extension as a good alternative to specialized databases for storing embeddings.** Researchers have shown it can be faster than Pinecone, cheaper, and is open-source, handling a good amount of operations (queries per second).

1. Load Data | 2. Query | 3. Exit
   Choice: 2
   Question: What is the difference between standard retrieval and the ColPali approach?

AI: Based on the provided context, there is no mention of a "ColPali approach."

However, the context describes a "simplest approach" for chunking, which can be considered a form of standard retrieval:

*   **Simplest Approach (Standard Retrieval):** This approach uses a standard size for chunks. The problem with this method is that chunks can break in the middle of a sentence or fail to preserve the main context needed for that chunk. This can lead to a much lower quality in the retrieval stage.
````
Due to some troubles with ColPali word recognition (Poly), I asked again in different words about the idea again at the end.

````
1. Load Data | 2. Query | 3. Exit
   Choice: 2
   Question: Why is hybrid search better than vector-only search?

AI: Hybrid search, which combines vector and keyword search, is better than vector-only search because it increases accuracy. According to the provided text, a hybrid system approach (vector plus keyword) increases accuracy by "a bit more than 10 percentage" compared to vector-only search.

1. Load Data | 2. Query | 3. Exit
   Choice: 2
   Question: what's the difference between standard rag and with ocr and vision skills?

AI: The main difference between standard RAG with OCR and RAG with vision skills (like the "poly library changes" approach) lies in how they process and understand information from documents, especially PDFs with complex visual elements:

1.  **Standard RAG with OCR:**
    *   **Process:** It utilizes Optical Character Recognition (OCR) to describe the information present on a PDF. This means OCR attempts to convert images, tables, and text into a textual description.
    *   **Vectorization:** This textual description is then vectorized and stored.
    *   **Limitations:** It often struggles with complex documents containing images, tables, and formulas, as OCR might not accurately describe these elements in a way that's easily parsable or fully understood by the LLM.

2.  **RAG with Vision Skills (e.g., Poly library changes):**
    *   **Process:** Instead of relying on OCR for description, this approach uses a **vision language model (VLM)**. It treats image representations (from PDFs) by splitting them into patches, which are then embedded directly into the model.
    *   **Retrieval:** When information is retrieved, it utilizes this embedded model and the visual language model to provide answers.
    *   **Advantages:** This method directly "understands" the visual content without needing to convert it to text via OCR first. It shows much better results in terms of information retrieval (approximately 15% better than standard OCR-based systems), especially for complex documents where OCR might fall short.

1. Load Data | 2. Query | 3. Exit
   Choice: 3
````