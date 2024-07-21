'''



def load_db(file, chain_type, k):
    # load documents
    #loader = PyPDFLoader(file)
    #documents = loader.load()
    #load all documents in directory
    loader = PyPDFDirectoryLoader(file)
    documents = loader.load() # documents is a list (#of list items = total # of pages of all documents in the directory, 
    #a list item = each page of the documents) 

    #add 'keyword' extracted from file name to metadata
    for test_doc in documents:
        print(f"original: {test_doc.metadata}")
        res = test_doc.metadata['source'].split(sep='\\')
        keyword = res[len(res)-1].split(sep='.pdf')[0]
        test_doc.metadata['keyword'] = keyword 
        print(f"new: {test_doc.metadata}")
    

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_size, chunk_overlap=Chunk_overlap) #chunk_size=1000, chunk_overlap=150
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
        verbose=True
    )
    return qa 

import param

class Cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(Cbfs, self).__init__( **params)
        self.panels = []
    
    def call_load_db(self, loaded_file):
        self.loaded_file = loaded_file
        self.qa = load_db(self.loaded_file, Chain_type, K) 
    
    def use_existing_db(self):
        embeddings2 = OpenAIEmbeddings()
        exdb = Chroma(persist_directory=persist_directory, embedding_function=embeddings2)
        # define retriever
        exretriever = exdb.as_retriever(search_type="similarity", search_kwargs={"k": K}) #k=4
        # create a chatbot chain. Memory is managed externally.
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0), 
            chain_type=Chain_type, #stuff
            retriever=exretriever, 
            return_source_documents=True,
            return_generated_question=True,
            verbose=True)

    def convchain(self, query):
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 







cb = Cbfs()

print(db_flag)
if db_flag == 0:  #1st time to embed the documents and save to the vector database
    loaded_file = "D:\Fai_personal\AI\Generative AI\MyProject\docs"
    print('embed documents and save to database: '+loaded_file)
    cb.call_load_db(loaded_file)
else:             #otherwise just set up QA chain from existing vector database
    print('use existing_db')
    cb.use_existing_db()


root=Tk()
root.title('Ask me')
root.iconbitmap('D:\Fai_personal\Graphics and audio\Little bunny boo boo\Bunnyicon.ico')

def retrieve_input():
    inputValue=textBox.get(1.0,END)
    cb.convchain(query=inputValue) 
    #clear textbox
    textBox.delete(1.0,END)
    #conversation = f"Question:\n{inputValue}Answer: No answer yet\n"
    #question = f"Question: {inputValue}"
    question = f"{inputValue} "
    Output.insert(END, question,'color')
    answer = f"    Answer: {cb.answer}\n"
    Output.insert(END, answer)
    ###########################new
    rOutput.insert(END, f"Reference: {question}  --> {cb.db_query}\n", 'color2')
    source = []
    for doc in cb.db_response:
        reference = f"File: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n"
        rOutput.insert(END, reference, 'color1')
        page_content = doc.page_content[0:500]
        rOutput.insert(END, page_content+'.......\n')
        source.append(doc) 
    

    

#Font
Font1 = ("Browallia New", 14)
Font2 = ("Helvetica", 10) 

#input textbox  
textBox=Text(root, height=5, width=65, foreground='black', background='white')
textBox.configure(font=Font1)
textBox.grid(row=1, column=0, pady=1)

#Enter button
buttonCommit=Button(root, height=7, width=4, text="Ask", foreground='white', background='#7171C6',command=lambda: retrieve_input())
buttonCommit.configure(font=Font2)
buttonCommit.grid(row=1, column=1, padx=2)

#output textbox with scrollbar
scrollbar = Scrollbar(root, width=20)
Output=Text(root, height=20, width=65, foreground='black', background='white',yscrollcommand = scrollbar.set)
Output.configure(font=Font1)
Output.grid(row=0, column=0, columnspan=1, padx=2)
Output.tag_configure('color', foreground='#6959CD', font=('Browallia New', 14,'bold'))
scrollbar.grid(row=0, column=1, sticky=NS)
#scrollbar.grid(row-1,column=3, rowspan=5, sticky=N+S)
scrollbar.config(command = Output.yview) 

#show greeting message
Output.insert(END, f"{greeting.content}\n")

#output reference document with schrollbar
rscrollbar = Scrollbar(root, width=20)
rOutput=Text(root, height=12, width=65, foreground='black', background='#ebebfb',yscrollcommand = rscrollbar.set) ##e4e4f7 #d8d8f0  #b4b4e1
rOutput.configure(font=Font1)
rOutput.grid(row=0, column=2, columnspan=1, rowspan=2, padx=2, sticky=NS)
rOutput.tag_configure('color1', foreground='#6959CD', font=('Browallia New', 14,'bold'))
rOutput.tag_configure('color2', foreground='#38bb4a', font=('Browallia New', 15,'bold'))
rscrollbar.grid(row=0, column=3, rowspan=2, sticky=NS)
#scrollbar.grid(row-1,column=3, rowspan=5, sticky=N+S)
rscrollbar.config(command = rOutput.yview) 


mainloop()


'''

###################################################################################################


from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import FastEmbedEmbeddings


from dotenv import load_dotenv
import os
import openai

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-4o"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
greeting = llm.invoke("Hello world!")
print(greeting.content)

########################variables###############################################
#persist_directory = 'D:\Fai_personal\AI\Generative AI\MyProject\docs\db' 
persist_directory =  './db/'
Chunk_size = 1000
Chunk_overlap = 150
K = 4
Chain_type = 'stuff'
db_flag  = 0   # 0: embed docuements and save to vector database, 1: use existing vector database
max_history_len = 10 #max number of latest chat history messages stored
session_id = "test123"
input_questions = ["what is the instructor of the machine learning class",
                   "what is the course number",
                   "who are the TAs", 
                   "Is programming skill required", 
                    ]
store={}

################################################################################

def embed_docs():
    loaded_file = "./docs/"
    print('embed documents and save to database: '+loaded_file)
    # load documents
    #loader = PyPDFLoader(file)
    #documents = loader.load()
    #load all documents in directory
    loader = PyPDFDirectoryLoader(loaded_file)
    documents = loader.load() # documents is a list (#of list items = total # of pages of all documents in the directory, 
    #a list item = each page of the documents) 

    #add 'keyword' extracted from file name to metadata
    for test_doc in documents:
        print(f"original: {test_doc.metadata}")
        res = test_doc.metadata['source'].split(sep='\\')
        keyword = res[len(res)-1].split(sep='.pdf')[0]
        test_doc.metadata['keyword'] = keyword 
        print(f"new: {test_doc.metadata}")
    

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_size, chunk_overlap=Chunk_overlap) #chunk_size=1000, chunk_overlap=150
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)


client = openai.OpenAI()
def get_completion(prompt, model=llm_name, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    if(len(store[session_id].messages) >= max_history_len):
        del store[session_id].messages[0:2]
    return store[session_id]

# Contextualize question into a standalone question based on chat history #
def create_standalone_question(input_question):
    contextualize_q_system_prompt2 = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt2),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    runnable = contextualize_q_prompt2 | llm
    
    standalone_question = runnable.invoke(
        {"input": input_question,
        "chat_history": get_session_history(session_id).messages}
        )
    return standalone_question

# Extract keywords from standalone question to be used for document retrieval filtering #
def extract_filter(standalone_question):
    extract_metadata_prompt = f"""extract all related keywords from the question below. Pick only the ones from the keyword list as follows. If you don't know what keyword to pick,
    answer as blank. 

    Keyword list: 
    - NT conference
    - Machine learning course
    - Intelligent operation center IOC
    - NT form
    - Smart pole

    Question: {standalone_question.content}

    Format the output in JSON as follows:
    question:<put the Question here>
    keyword:<put all related keywords here delimited by , as shown in example below or don't put anything here if no keyword in the Keyword list found>

    Example for keyword output:
    keyword: keyword1, keyword2, keyword3,...
    """

    response = get_completion(extract_metadata_prompt, model=llm_name, temperature=0.0)
    #print(response)

    # Parse LLM output and create a filter dictionary
    question_schema = ResponseSchema(name="question",
                                description="standalone question to be sent to another LLM call.")
    keyword_schema = ResponseSchema(name="keyword",
                                description="a list of keywords of the relavant document extracted from the standalone question.")

    response_schemas = [question_schema, 
                        keyword_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_dict = output_parser.parse(response)

    print(output_dict)


    # list of keywords extracted from the LLM output 
    list_parser = CommaSeparatedListOutputParser()
    lst = list_parser.parse(output_dict["keyword"]) 

    # Create a filter dictionary to filter by keywords
    filter_dict = {"keyword": {"$in": lst}}
    #print(filter_dict)
    return filter_dict

# Create a retriever that filters the documents using filter dictionary (related keywords) #
def create_retrieval(K,filter_dict):   
    fembeddings = OpenAIEmbeddings()
    fdb = Chroma(persist_directory=persist_directory, embedding_function=fembeddings)
    fretriever = fdb.as_retriever(search_type="similarity", search_kwargs={"k": K, 'filter':filter_dict})
    print(persist_directory)
    return fretriever







class AskMe:
    db = None
    retriever = None
    chain = None
       
    
    def ask(self, input_question: str):
        
        standalone_question = create_standalone_question(input_question)
        filter_dict = extract_filter(standalone_question)
        fretriever = create_retrieval(K,filter_dict)


        # Answer question #
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \

        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(fretriever, question_answer_chain)


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        #print("orginal question: "+input_question)
        #print("standalone question: "+standalone_question.content)
        response = conversational_rag_chain.invoke(
            {"input": standalone_question.content,
            "chat_history": {}},
            config={
                "configurable": {"session_id": session_id}
            }
        )
        
        #greeting = llm.invoke(input_question)
        #return greeting.content
        #return response'''
        
        return response 

    def clear(self):
        self.db = None
        self.retriever = None
        self.chain = None

print(store)
#embed_docs()
#print('Finished embedding docs')
#a = AskMe()
#for item in input_questions:
#    print("original question: "+item)
#    print(a.ask(item)['answer'])
#    print('-----------------------------------------')
print(store)