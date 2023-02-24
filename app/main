from flask import Flask, render_template, request

#from chatbot import chatbot
import os
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient 
from azure.search.documents import SearchClient
import openai
import re
import requests
import sys
from num2words import num2words
import os
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from azure.ai.textanalytics import TextAnalyticsClient


import os
#setting up Azure cognitive service
service_name = "searchsj" # Cognitive Search Service Name
admin_key = "NjjULpqs4jLMKUujKvgPL0vnOPsMRqoScxyz37q2NNAzSeBMg4F3" # Cognitive Search Admin Key
index_name = "car1" # Cognitive Search index name

# Create an SDK client
endpoint = "https://searchsj.search.windows.net"
credential = AzureKeyCredential(admin_key)

search_client = SearchClient(endpoint=endpoint,index_name=index_name,api_version="2021-04-30-Preview",credential=credential)
API_KEY = "" # SET YOUR OWN API KEY HERE
RESOURCE_ENDPOINT = "" # SET A LINK TO YOUR RESOURCE ENDPOINT
openai.api_type = "azure"
openai.api_key = "1100f07809eb4982b75f7643392c0c3c"
openai.api_base = "https://sjaoai.openai.azure.com/"
openai.api_version = "2022-12-01" #openai api version m

TEXT_SEARCH_DOC_EMBEDDING_ENGINE = 'searchdoc01' # Model deployment name - mentioned in the above screenshot 
TEXT_SEARCH_QUERY_EMBEDDING_ENGINE = 'searchquery01' # Model deployment name - mentioned in the above screenshot
TEXT_DAVINCI_001 = "textdavinci02" # Model deployment name - mentioned in the above screenshot

text_endpoint = "https://sj-l.cognitiveservices.azure.com/"
text_key = "c2d1304572cc40ffb3f50f2bcec24de4"
project_name = "language-bank"
deployment_name = "language-bank"

 # Training with Personal Ques & Ans 
text_analytics_client = TextAnalyticsClient(endpoint=text_endpoint,credential=AzureKeyCredential(text_key))


#trainer = ListTrainer(chatbot)
#trainer.train(training_data) 
# Training with English Corpus Data 
#trainer_corpus = ChatterBotCorpusTrainer(chatbot)
#Defining helper functions
#Splits text after sentences ending in a period. Combines n sentences per chunk.
def splitter(n, s):
    pieces = s.split(". ")
    list_out = [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]
    return list_out

# Perform light data cleaning (removing redudant whitespace and cleaning up punctuation)
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def search_docs(df, user_query, top_n=3):
    embedding = get_embedding(
        user_query,
        engine=TEXT_SEARCH_QUERY_EMBEDDING_ENGINE
    )
    df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )
    return res
    
app = Flask(__name__)
app.static_folder = 'static'

    
@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    poller = text_analytics_client.begin_single_label_classify([userText],project_name=project_name,deployment_name=deployment_name)
    document_results = poller.result()
    response = text_analytics_client.recognize_entities([userText], language="en")
    result = [doc for doc in response if not doc.is_error]
    cat=""
    score=0
    for doc in result:
        for entity in doc.entities:
            score=entity.category
    print(score)
    if score=="Product":
        results = search_client.search(search_text=userText, include_total_count=True)
        document = next(results)['review']
        document_chunks = splitter(10, normalize_text(document))
        embed_df = pd.DataFrame(document_chunks, columns = ["chunks"])
        embed_df['curie_search'] = embed_df["chunks"].apply(lambda x : get_embedding(x, engine = TEXT_SEARCH_DOC_EMBEDDING_ENGINE))
        #document_specific_query = "trouble in clinton campaign" 
        res = search_docs(embed_df, userText, top_n=2)
        result_1 = res.chunks[0]
        #result_2 = res.chunks[1]
        header = "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say I don't know"
        #question="what is the trouble so far with clinton campaign"
        chosen_sections=normalize_text(result_1)
        prompt_i = header + "".join(chosen_sections) + "\n\n Q: " + userText + "\n A:"

        # Using a low temperature to limit the creativity in the response. 
        response = openai.Completion.create(
                engine= "textdavinci02",
                prompt = prompt_i,
                temperature = 0.0,
                max_tokens = 500,
                top_p = 1.0,
                frequency_penalty=0.5,
                presence_penalty = 0.5,
                best_of = 1
                
            )
    else:
        # Using a low temperature to limit the creativity in the response. 
        print("else")
        header2="Respond to the text below in a way you are chit chatting with a human."
        prompt_j=header2+ "\n\n Q: " + userText + "\n A:"
        response = openai.Completion.create(
                engine= "textdavinci02",
                prompt = prompt_j,
                temperature = 0.5,
                max_tokens = 60,
                top_p = 0.5,
                frequency_penalty=0,
                presence_penalty = 0,
                best_of = 1
                
            )
        print(response)
        

    #print(response.choices[0].text)
    return str(response.choices[0].text)


