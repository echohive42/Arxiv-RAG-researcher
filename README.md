# ArXiv RAG Researcher with GPT-4o

This tool allows you to search ArXiv for scientific papers, extract their content, embed and chunk the text, and ask questions about them using GPT-4o.

#### You can find 250+ projects like this at my Patreon, where I also offer consulting: https://www.patreon.com/echohive42.
#### You can find all my videos about building LLM powered apps at my website https://www.echohive.live. Or at my YouTube channel https://www.youtube.com/@echohive.

### Give this repo a star if you find it helpful 🙏

[Watch the demo video](https://youtu.be/cnqIomsS-SU)

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

1. Run the script:
   ```
   python arxiv-rag-researcher.py
   ```

2. Follow the prompts:
   - Choose a search mode (relevance or submitted date)
   - Enter the number of papers to search
   - Enter your search query (or 'skip' to use an existing collection)

3. The tool will:
   - Download the PDFs of the selected papers
   - Extract text from the PDFs
   - Chunk the text with a 1/3 page overlap for context preservation
   - Embed the chunks using OpenAI's text-embedding-3-large model
   - Store the embedded chunks in a new Chroma DB collection for each search

4. Once papers are processed, you can ask questions about them:
   - Enter your question
   - Specify the number of relevant chunks to retrieve

5. The tool will:
   - Query the current Chroma DB collection to find the most relevant chunks
   - Display these chunks from the papers
   - Provide an answer using GPT-4o based on the retrieved chunks

6. Questions and answers are saved in `qa_history.json` for future reference.

7. Type 'quit' to exit the question-answering loop and start a new search, or exit the program.

## Multiple Collections and Querying

- Each search creates a new Chroma DB collection, allowing you to maintain separate sets of papers for different topics or searches.
- You can skip the search process and query existing collections:
  - Enter 'skip' when prompted for a search query
  - The tool will list available collections
  - Choose a collection to query from the list
- This feature allows you to switch between different paper sets without re-downloading or re-processing papers.

Note: This tool uses Chroma DB for efficient storage and retrieval of embedded paper chunks, allowing for semantic search capabilities across multiple collections.
