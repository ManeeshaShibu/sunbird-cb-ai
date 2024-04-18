from flask import Flask, request, jsonify
import os
import spacy
import re
import PyPDF2
import torch.nn.functional as F
from modules.coref_resolver import coref_impl
from repo.milvus_entity import milvus_collection
from modules.text_processor import process_text
from modules.faq_handler import faq
import pandas as pd
import json
from modules.generative_model import answer_generation
from sentence_transformers import SentenceTransformer
import gc

app = Flask(__name__)

nlp_model = spacy.load("en_core_web_sm")
transformer_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
text_processor_preloaded = process_text(nlp_model, transformer_model)
faq_obj = faq()
milvus = milvus_collection()
print("start")
CONF = None
with open('conf/config.json') as config_file:
    CONF = json.load(config_file)

# Create 'upload_folder' directory if it doesn't exist
if not os.path.exists('upload_folder'):
    os.makedirs('upload_folder')

coref_obj = coref_impl()

generate_answer = answer_generation()

@app.route('/')
def index():
    return 'Welcome to the PDF Ingestion API!'

@app.route('/drop/collecion', methods = ['POST'])
def drop_collection():
    data = request.get_json()
    collection = data.get('collection_name', '')
    return milvus.drop_collection(collection)

@app.route('/get/collecions')
def get_collection():
    return milvus.get_collections()

@app.route('/uploader', methods = ['POST'])
def upload_file():
    text_processor = process_text(nlp_model, transformer_model)
    if request.method == 'POST':
        file = request.files['file']
    collection = milvus.get_collection()


    if file:
        # print(file)
        uploaded_file_path = os.path.join("upload_folder", file.filename)
        # print(uploaded_file_path)
        file.save((uploaded_file_path))
        # print('staged file locally: ' + str(file.filename))
        text_list = []
        embedding_list = []
        page_list = []
        # Extract text and page from the PDF
        if file.filename.endswith(".pdf"):
            text_list, embedding_list, page_list, doc_list, doc_parent_list = text_processor.extract_text_from_pdf(uploaded_file_path)
        elif file.filename.endswith(".mp4"):
            # print("@@@@@@@@@@@")
            text_list, embedding_list, page_list, doc_list, doc_parent_list = text_processor.extract_text_from_video(file.filename,uploaded_file_path)
        # Insert data into Milvus collection
        # print('inseritng into collection')
        #    schema = CollectionSchema(fields=[document_id, page, page_page, embeddings, text], enable_dynamic_field=True)
        df = pd.DataFrame()
        df['text_list'] = text_list
        df.to_csv('check_text.csv')
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(len(embedding_list))
        print(text_list)
        print(page_list)
        print(doc_list)
        print(doc_parent_list)
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")
        
        collection.insert([ embedding_list, text_list, page_list, doc_list, doc_parent_list])
        
        # Create an index on the "embeddings" field
        index_params = {
            'metric_type': 'L2',
            'index_type': "HNSW",
            'efConstruction': 40,
            'M': 20
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        print('Index created.')

        #loads collection if not loaded already
        milvus.load_collection()
        #remove staged file - add try except for all I/O
        os.remove(uploaded_file_path)
        del file
        del text_processor
        del embedding_list
        del text_list
        del page_list
        del collection
        del doc_list
        del doc_parent_list
        
        gc.collect()

        return jsonify({'message': 'Data inserted into the collection.'}), 200

    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/ingest/faq', methods = ['POST'])
def ingest_content():
    data = request.get_json()
    print(data)
    faqs =  data.get('faqs', '')
    faq_obj.load_faq(faqs)
    return jsonify({'message': 'loaded the FAQs'}), 200
    

@app.route('/search-answers', methods=['POST'])
def search_answers():
    data = request.get_json()
    print(data)
    collection_name = os.getenv('milvus_collection_name', CONF["milvus_collection_name"])
    query = data.get('query', '')
    print(collection_name)
    print(query)
    clean_query = text_processor_preloaded.clean_text(query)    

    faq_response, score = faq_obj.query(clean_query)
    if faq_response and score > float(os.getenv('faq_cutoff_direct', CONF["faq_cutoff_direct"])):
        return faq_response, 200     

    query_encode = text_processor_preloaded.get_model().encode(clean_query)
    collection = milvus.get_collection()
    collection.load()
    print("Collection loaded.")

    # Encode the query
    # Perform a search to get answers    
    search_results = collection.search(data=[query_encode], anns_field="embeddings",
                                      param={"metric": "L2", "offset": 0},
                                      output_fields=["page", "page_page", "text", "doc", "doc_parent"],
                                      limit=int(os.getenv('milvus_top_n_results', CONF["milvus_top_n_results"])), consistency_level="Strong") 
    print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLl")   
    print(search_results)
    # Extract relevant information from search results
    answers_final = []
    for result in search_results:
        for r in result:
            answers_final.append({"text-chunk" : r.entity.text, "similarity_distacne" : r.distance, "do_id" : r.entity.doc, "Page" : r.entity.page, "document": r.entity.doc_parent})
    return jsonify({'answers_final': answers_final}), 200

@app.route('/generate-answers', methods=['POST'])
def generate_answers():
    data = request.get_json()
    print(data)
    #collection_name = data.get('collection_name', '')
    collection_name = os.getenv('milvus_collection_name', CONF["milvus_collection_name"])
    query = data.get('query', '')
    clean_query = text_processor_preloaded.clean_text(query)
    faq_response, faq_score = faq_obj.query(clean_query)

    if faq_response and faq_score > float(os.getenv('faq_cutoff_direct', CONF["faq_cutoff_direct"])):
        return faq_response, 200 
    
    print(collection_name)
    print(query)
    # Define and load the Milvus collection
    collection = milvus.get_collection()
    collection.load()
    print("Collections loaded.")

    # Encode the query
    query_encode = text_processor_preloaded.get_model().encode(query.lower())

    # Perform a search to get answers
    search_results = collection.search(data=[query_encode], anns_field="embeddings",
                                      param={"metric": "L2", "offset": 0},
                                      output_fields=["page", "page_page", "text"],
                                      limit=10, consistency_level="Strong")
    
    
    answers_final = []
    for result in search_results:
        for r in result:
            answers_final.append({"text-chunk" : r.entity.text, "similarity_distacne" : r.distance, "do_id" : r.entity.doc, "Page" : r.entity.page, "document": r.entity.doc_parent})

    answers_final = sorted(answers_final, key=lambda x: x["similarity_distacne"], reverse=False)
    top_n = int(os.getenv('top_matching_chunks_as_context', CONF["top_matching_chunks_as_context"]))
    context = ""
    for answer in answers_final[:top_n]:
        print(answer['text-chunk'])
        context = context + "          " + answer['text-chunk']
    
    if faq_score > float(os.getenv('faq_cutoff_overall', CONF["faq_cutoff_overall"])):
        context = context + "          " + faq_response['generated_ans']["text-chunk"] + " " + faq_response['generated_ans']["answer"]["answer"]

    #print(search_results)
   
    # Extract relevant information from search results
    #answers_final = '               '.join(answers_final)
    generated_ans = generate_answer.openai_answer(query,context)

    return jsonify({'generated_ans': generated_ans, 'closest context' : answers_final[:top_n]}), 200
if __name__ == '__main__':
    print("running on 5000 port")
    app.run(debug=False, host='0.0.0.0', port=5000)

