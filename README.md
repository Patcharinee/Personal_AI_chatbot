# 🤖  Personal AI Chatbot
This generative AI chatbot is designed to answer questions both about your documents and a wide range of other topics.

Users can interact with the chatbot in a natural, conversational manner. It maintains context throughout the discussion, allowing for more coherent and relevant responses.

When a question relates to documents that have been embedded and stored in the vector database, the chatbot retrieves information from those sources and provides a response with references to the relevant documents.

If the question isn’t related to any stored documents, the chatbot uses its LLM capabilities to generate a response without referencing the document database.

# Features
- Use your own PDF documents as a source of information that the chatbot can query in addition to its built-in LLM capabilities.
- The system reformulates user queries into standalone questions, extracts relevant keywords, and retrieves the most related information from your documents.
- An interactive chat UI built using Streamlit allows for a user-friendly interaction with the AI.
- Keeps a history of interactions up to a defined limit for contextual question answering.

# How It Works
### Embedding Documents: 
Load your PDF documents into the "docs" folder and click "Embed" to embed the documents using the OpenAI embedding models and storing the resulting data in a Chroma vector database.

### Keyword Extraction: 
The keywords of the documents are automatically extracted from the filenames and stored in the file "keyword_list.txt". 

### Asking Questions:
Each question is first reformulated to be standalone.
The system then extracts keywords from the question and filters the documents using these keywords.
A retrieval process is used to fetch context from the filtered documents which is then sent to the LLM to answer the question.

### Integration:

- Utilizes OpenAI's GPT models for language understanding.
- Supports embedding and retrieval using the Langchain library mechanisms.
