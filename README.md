# RAG chatbot
This is one of the capstone project of my AI engineering course. I want to leave my work here to present my understanding. The code can be viewed in my repositories (main.py). 

# Setup
These are the packages that are needed.
```
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import gradio as gr
from dotenv import load_dotenv
```

Note that the versions are:
```
gradio==6.1.0
langchain-classic==1.0.0
langchain-community==0.4.1
langchain-google-genai==4.0.0
langchain-text-splitters==1.0.0
```

It is alright to use other LLM and Embedding models. Google's models are used here as an example. Alternatives maybe huggingface or Mistral.

Remember to use ```load_dotenv()``` to load the API key from environment(.env).

# Process

The whole process can be broken down a few parts. ```PyPDFLoader``` first load the document (pdf for this bot), and then (```RecursiveCharacterTextSplitter```) split it into chunks into acceptable size for the embedding model (```GoogleGenerativeAIEmbeddings```) to run. This will convert chunks into context vectors and then store them into vector database (```Chroma```).

Finally, we will utilize LLM to understand our query and use ```RetrievalQA``` to look up the context vectors that are most likely to answer the query from the source (pdf).

# Interface

To wrap all these things up, we can host a website to give a more user-friendly interface to run the process by gradio.

```
# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="file"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)
```

```fn``` represent the function we mentioned before for the RAG process. It accepts inputs from user's upload and query(prompt) from user's text input, and then pass the inputs to function and return the output. 

And now we are ready to launch.
```
rag_application.launch(server_name="127.0.0.1", server_port= 7890)
```

It should look like this.

<img width="1518" height="452" alt="image" src="https://github.com/user-attachments/assets/943a66a0-2f31-41b2-8167-3362dca887c8" />

# Follow up

There sereval things can be improved that I would like to address.

LLM configuration: I don't suggest to tune the ```temperature```, but one may change ```top_k``` and ```top_k``` and so on.

Multiple input: Extend from one pdf to more is definitely worth trying. It can be various file type, or many pdfs, or both. However, this requires complete update on the code syntax.

Refine QA quality: One can further improve the reponse from LLM by creating a prompt template ```custom_prompt = PromptTemplate.from_template(custom_prompt_template)```, and then pass to the ```chain_type_kwargs={"prompt": custom_prompt}``` inside the argument of ```RetrievalQA.from_chain_type```. This is helpful because this fits better what a user wants by giving a better context, such as not to answer something not in the document.
