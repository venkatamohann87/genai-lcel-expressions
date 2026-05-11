# Design and Implementation of LangChain Expression Language (LCEL) Expressions

## AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

## PROBLEM STATEMENT: 
Design an LCEL pipeline using LangChain with at least two dynamic prompt parameters.  Integrate prompt, model, and output parser components to form a complete expression.  Evaluate its functionality through real-world query-response scenarios.

## DESIGN STEPS:

#### STEP 1: Setup API and Environment: Load environment variables using dotenv and set openai.api_key from the local environment.

#### STEP 2: Create Prompt and Model: Use LangChain to define a ChatPromptTemplate and initialize ChatOpenAI for text generation.

#### STEP 3: Build a Retrieval System: Store predefined texts in DocArrayInMemorySearch with OpenAIEmbeddings and create a retriever.

#### STEP 4: Define Question-Answering Chain: Use RunnableMap to fetch relevant documents and pass them to a chat model for responses.

#### STEP 5: Invoke the Chain: Run chain.invoke() with a question to retrieve context-based answers using the LangChain pipeline.

## PROGRAM:
<h3> Name: Venkata Mohan N</h3>
<H3> Reg NO: 212224230298</H3>

### Simple Chain
```py

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "tell me about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "one piece"})
```


### Complex Chain
```py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch   

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Generative AI is a type of artificial intelligence that creates new content, such as text, images, or code, based on learned patterns.","(LCEL)LangChain Expression Language (LCEL) is a declarative syntax for composing and chaining LangChain components efficiently."],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("what is generative ai?")
retriever.get_relevant_documents("what is the full form of LCEL")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "what is the full form of LCEL?"})

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "what is the full form of LCEL?"})
```


## OUTPUT:
### Simple Chain:

<img width="1141" height="445" alt="image" src="https://github.com/user-attachments/assets/8b4c26cc-8c73-4ddb-85af-87469dd4d1fb" />


### Complex Chain:

<img width="1233" height="827" alt="image" src="https://github.com/user-attachments/assets/96aa43a2-cc2f-4c2e-99e7-1ebb9f51df53" />

## RESULT: 
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
