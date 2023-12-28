# Implementation of RAG with LLAMA-2 🦙, Hugging-Face 🤗 ,and LangChain ⛓️

<ul>
  <li>bellow is the complete architecture of RAG</li>
  <li>We will implement it with <b> LangChain and Hugging-Face LLama-2 </b></li>
</ul>

![RAG](https://github.com/LLama2-Ai/RAG-with-LLAMA-2---Langchain/assets/142317270/4bcaa4de-f256-4f01-abd7-032501896d9f)


<h6>Install the dependences first</h6>

```
! pip install langchain
! pip install chromadb
! pip install sentence-transformers
```

<ul>
  <li> <b>Lang chain</b> has the built-in classes to implement the RAG like :: <b>RetrievalQA</b> </li>
  <li> <b>Chroma</b> db will be used to save the embeddings for the new docs </li>
  <li><b>sentence-transformers</b> db will be used to create the embeddings for the new docs </li>
</ul>

<br/>

```
import transformers as t
from langchain.chains import RetrievalQA
from transformers import TextStreamer 
```
<ul>
  <li> <b>transformers</b> To download the llama-2 from hugging-face. but you need to get the access key for it as it is a gated model. </li>
  <li> <b>RetrievalQA</b>This chain will manage the complete RAG pipeline </li>
  <li><b>TextStreamer</b> Will be used with model.generate method to decode the tokens back to words. </li>
</ul>

</br>

<h6>Load the llama-2 from hugging face</h6>

```
model_id='meta-llama/Llama-2-7b-chat-hf'
model=t.AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf',token='your_hugging_face_API_KEY')

tokenizer = t.AutoTokenizer.from_pretrained(model_id,token='your_hugging_face_API_KEY')
```

<h5>Lets test the model</h5>

```
prompt='''
### User: what is the fastest car in the world and how much it costs
### Assistant:
'''
inputs=tokenizer(prompt,return_tensors='pt')
streamer =TextStreamer(tokenizer=tokenizer,skip_prompt=True,skip_special_tokens=True)
res=model.generate(**inputs,streamer=streamer,use_cache=True)

```
![image](https://github.com/LLama2-Ai/RAG-with-LLAMA-2---Langchain/assets/142317270/957076ca-aced-4591-b5b8-570a1e086a54)

<h6>Now lets implement the RAG</h6>
<h2>RAG</h2>

```
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain import HuggingFacePipeline 
from transformers import pipeline 
from langchain import PromptTemplate 
```
<ul>
  <li><b>PyPDFLoader,DirectoryLoader</b>  Will help to read all the files from a directory </li>
  <li><b> HuggingFaceEmbeddings </b>  Will be used to load the sentence-transformer model into the LangChain. </li>
  <li><b> RecursiveCharacterTextSplitter </b>  Used to split the docs and make it ready for the embeddings. </li>
  <li><b> HuggingFacePipeline </b> It will convert the hugging-face model to LangChain relevant llm. </li>
  <li><b> pipeline </b> will couple up the model and tokenizer together and allow us to set some attributes as well. </li>
  <li><b> PromptTemplate </b>   will be used to create custom prompt. </li>
</ul>


```
loader=DirectoryLoader('./CS204_Slides_1_15/',glob='*.PDF',loader_cls=PyPDFLoader)
documents=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings =HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2',
                                      model_kwargs={'device':'cpu'})

```

```
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings,
                                 persist_directory='db/')
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory='db/', 
                  embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

```

```
custom_prompt_template="""
kindly guide the user on following query 
Context:{context}
Question:{question}
Only return the relevant answer
"""
prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])
```
```
question_answerer = pipeline(
    task="text-generation",
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True
)
# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm2 = HuggingFacePipeline(
    pipeline=question_answerer
)

```

<h2>Lets build the final chain</h2>

```
qa_chain = RetrievalQA.from_chain_type(llm=llm2, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)


res=qa_chain('tell me the topics in cyber society')
print(res['result'])

```


![image](https://github.com/LLama2-Ai/RAG-with-LLAMA-2---Langchain/assets/142317270/b07c44d7-f720-478e-8cbe-32b9f2e6f232)



