from langchain.document_loaders import PyPDFLoader
from langchain                  import PromptTemplate, LLMChain
from langchain.embeddings       import HuggingFaceEmbeddings
from langchain.llms             import GPT4All

from langchain.text_splitter              import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss         import FAISS
from langchain.callbacks.base             import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

documents       = PyPDFLoader('path to your pdf').load_and_split()
text_splitter   = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=64)
texts           = text_splitter.split_documents(documents)
embeddings      = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
faiss_index     = FAISS.from_documents(texts, embeddings)

faiss_index.save_local("path to folder where you want to store index")

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# load vector store
print("loading indexes")
faiss_index = FAISS.load_local("path to your index folder", embeddings)
print("index loaded")
gpt4all_path = 'path to your llm bin file'

# # Set your query here manually
question = "your query"
matched_docs = faiss_index.similarity_search(question, 4)
context = ""
for doc in matched_docs:
    context = context + doc.page_content + " \n\n "

template = """
Please use the following context to answer questions.
Context: {context}
 - -
Question: {question}
Answer: Let's think step by step."""

callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
llm              = GPT4All(model=gpt4all_path,n_ctx=1000, callback_manager=callback_manager, verbose=True,repeat_last_n=0)
prompt           = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain        = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))