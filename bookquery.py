


import os
from dotenv import load_dotenv

from call_rag_system import rag_system


load_dotenv()



base_url = "https://books.toscrape.com/"
google_api_key = os.getenv("GEMINI_API_KEY")

rag_system = rag_system(base_url, google_api_key=google_api_key)

while True:
    query = input("Enter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    response = rag_system({"question": query})
    print(response["answer"])
    print("----------------------------------------------------")