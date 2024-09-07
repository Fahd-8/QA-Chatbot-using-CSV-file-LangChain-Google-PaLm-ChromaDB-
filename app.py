from flask import Flask, request, render_template
import pickle
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)

from langchain_community.llms import GooglePalm

api_key="AIzaSyAuRMpuPK9FdeT2WTEmcIupA98VQSZyvoE"

llm= GooglePalm(google_api_key=api_key, temperature=0.7)

# Load the vector store
with open('vectordb.pkl', 'rb') as f:
    vectordb = pickle.load(f)

# Initialize the retriever and QA chain
retriever = vectordb.as_retriever()

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = qa_chain(question)
    return render_template('index.html', question=question, answer=answer['result'])

if __name__ == '__main__':
    app.run(debug=True)
