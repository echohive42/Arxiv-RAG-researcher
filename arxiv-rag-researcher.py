import arxiv
import aiohttp
import asyncio
import PyPDF2
import io
import json
from asyncio import Semaphore
import chromadb
from chromadb.utils import embedding_functions
import os
from openai import OpenAI
from termcolor import colored


# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# Initialize OpenAI client
openai_client = OpenAI()

async def fetch_pdf(session, url, semaphore):
    async with semaphore:
        async with session.get(url) as response:
            return await response.read()

async def extract_text_from_page(page):
    return page.extract_text()

async def extract_text_from_pdf(pdf_content):
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    async def process_page(i):
        current_page = await extract_text_from_page(pdf_reader.pages[i])
        if i == 0:
            return current_page
        else:
            previous_page = await extract_text_from_page(pdf_reader.pages[i-1])
            overlap = len(previous_page) // 3
            return previous_page[-overlap:] + current_page

    tasks = [asyncio.create_task(process_page(i)) for i in range(len(pdf_reader.pages))]
    texts = await asyncio.gather(*tasks)
    return texts

async def search_arxiv(user_input, search_mode, n):
    try:
        # Create a new collection for each search
        collection_name = f"arxiv_search_{user_input.replace(' ', '_')}"[:50]
        collection = client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

        if search_mode == 'relevance':
            results = list(arxiv.Search(query=user_input, max_results=n, sort_by=arxiv.SortCriterion.Relevance).results())
        elif search_mode == 'latest':
            results = list(arxiv.Search(query=user_input, max_results=n, sort_by=arxiv.SortCriterion.LastUpdatedDate).results())
        else:
            print("Invalid search mode. Please enter 'relevance' or 'latest'.")
            return None

        if not results:
            print("No results found for your query.")
            return None
        else:
            print(f"Results for '{search_mode}':")
            semaphore = Semaphore(100)
            async with aiohttp.ClientSession() as session:
                tasks = [asyncio.create_task(fetch_pdf(session, result.pdf_url, semaphore)) for result in results]
                pdf_contents = await asyncio.gather(*tasks)

                async def process_paper(i, result, pdf_content):
                    pages = await extract_text_from_pdf(pdf_content)
                    
                    paper_data = {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "summary": result.summary,
                        "url": result.pdf_url
                    }
                    
                    # Add to Chroma collection
                    collection.add(
                        documents=pages,
                        metadatas=[{"title": result.title, "page": j} for j in range(len(pages))],
                        ids=[f"paper_{i+1}_page_{j}" for j in range(len(pages))]
                    )
                    
                    # Save paper data to JSON file
                    json_file = "paper_metadata.json"
                    if os.path.exists(json_file):
                        with open(json_file, "r+") as file:
                            data = json.load(file)
                            data.append(paper_data)
                            file.seek(0)
                            json.dump(data, file, indent=4)
                    else:
                        with open(json_file, "w") as file:
                            json.dump([paper_data], file, indent=4)
                    
                    print(f"Title: {result.title}")
                    print(f"Authors: {', '.join(author.name for author in result.authors)}")
                    print(f"Summary: {result.summary[:100]}...")
                    print("URL:", result.pdf_url)
                    print(f"Added to Chroma collection: {collection_name}")
                    print(f"Metadata saved to: {json_file}")
                    print("-" * 20)

                await asyncio.gather(*[process_paper(i, result, pdf_content) for i, (result, pdf_content) in enumerate(zip(results, pdf_contents))])
        
        return collection_name
    except arxiv.ArxivError as e:
        print(f"An error occurred while searching arXiv: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the PDFs: {e}")
        return None

print("Welcome to the ArXiv Search Tool!")

async def main():
    collection_name = None
    while True:
        user_input = input(colored("Enter your search query to search for arxiv papers (or 'skip' to skip): ", "green"))
        if user_input.lower() == 'skip':
            break
        
        print(colored("Choose a search mode:", "blue"))
        print(colored("1. Relevance", "green"))
        print(colored("2. Submitted Date", "green"))
        choice = input("Enter the number of your choice: ")
        
        if choice == "1":
            search_mode = "relevance"
        elif choice == "2":
            search_mode = "latest"
        else:
            print("Invalid choice. Defaulting to relevance.")
            search_mode = "relevance"
        
        n = int(input(colored("Enter the number of papers to search: ", "blue")))
        
        collection_name = await search_arxiv(user_input, search_mode, n)
        if collection_name:
            break

    # Enter chat loop
    while True:
        query = input(colored("Enter your question to get answers with rag and gpt-4o (or 'quit' to exit): ", "yellow"))
        if query.lower() == 'quit':
            break
        
        k = int(input("Enter the number of chunks to retrieve: "))
        
        # List collections if no collection name is given
        if not collection_name:
            collections = client.list_collections()
            print("\nAvailable collections:")
            for i, coll in enumerate(collections):
                print(f"{i+1}. {coll.name}")
            choice = int(input("Choose a collection number: ")) - 1
            collection_name = collections[choice].name
        
        collection = client.get_collection(name=collection_name, embedding_function=openai_ef)
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        print("\nRelevant chunks:")
        chunks = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\nChunk {i+1}:")
            print(f"From: {metadata['title']}, Page: {metadata['page']}")
            print(doc[:500] + "...")  # Print first 500 characters of each chunk
            chunks.append(f"Chunk {i+1} from {metadata['title']}, Page {metadata['page']}: {doc}")

        # Prepare the prompt for GPT-4
        prompt = f"Based on the following chunks of information from scientific papers, please answer this question: {query}\n\n"
        prompt += "\n\n".join(chunks)

        # Send to GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on scientific paper excerpts."},
                {"role": "user", "content": prompt}
            ]
        )

        # Print the response
        print("\nGPT-4 Response:")
        print(response.choices[0].message.content)

        # Append question and response to JSON file
        qa_entry = {
            "question": query,
            "answer": response.choices[0].message.content
        }
        
        json_file = "qa_history.json"
        if os.path.exists(json_file):
            with open(json_file, "r+") as file:
                data = json.load(file)
                data.append(qa_entry)
                file.seek(0)
                json.dump(data, file, indent=4)
        else:
            with open(json_file, "w") as file:
                json.dump([qa_entry], file, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
