import os
import signal
import sys
from langdetect import detect

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

GEMEINI_API_KEY = "AIzaSyD6WAD6wDXirnBOg8ua3GP5pkvRBJ7Loz8"

def signal_handler(sig, frame):
    print('\nThanks for using Gemini. :)')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def generate_rag_prompt(query, context, language):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    if language == 'ar':
        prompt = ("""
أنت روبوت مفيد ومطلع يجيب على الأسئلة باستخدام النص من سياق المرجع المدرج أدناه. تأكد من الإجابة بجملة كاملة، كونك شاملًا، بما في ذلك جميع المعلومات الأساسية ذات الصلة. ومع ذلك، أنت تتحدث إلى جمهور غير تقني، لذا تأكد من تبسيط المفاهيم المعقدة واعتماد نبرة ودية وحوارية. إذا كان السياق غير ذي صلة بالإجابة، يمكنك تجاهله.
                السؤال: '{query}'
                السياق: '{context}'
              
              الإجابة:
              """).format(query=query, context=context)
    else:
        prompt = ("""
You are a helpful and informative bot that answers questions using text from the reference context included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and conversational tone. \
  If the context is irrelevant to the answer, you may ignore it.\
                QUESTION: '{query}'
                CONTEXT: '{context}'
              
              ANSWER:
              """).format(query=query, context=context)
    return prompt

def get_relevant_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    genai.configure(api_key=GEMEINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

welcome_text = generate_answer("Can you quickly introduce yourself")
print(welcome_text)

while True:
    print("-----------------------------------------------------------------------\n")
    print("What would you like to ask?")
    query = input("Query: ")
    language = detect(query)
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context, language=language)
    answer = generate_answer(prompt=prompt)
    print(answer)
