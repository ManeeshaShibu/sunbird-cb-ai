#### Pdf ingestion API
# http://127.0.0.1:5000//upload-pdf/<collection_name>
# payload format body-form_data-key(file(default))- value(.pdf file path)


from flask import Flask, request, jsonify
import os
import spacy
import re
import PyPDF2
import torch.nn.functional as F
from modules.coref_resolver import coref_impl
from repo.milvus_entity import milvus_collection
from modules.text_processor import process_text
import pandas as pd
import json
from modules.generative_model import answer_generation

app = Flask(__name__)

milvus = milvus_collection()
text_processor = process_text()
CONF = None
with open('conf/config.json') as config_file:
    CONF = json.load(config_file)

# Create 'upload_folder' directory if it doesn't exist
if not os.path.exists('upload_folder'):
    os.makedirs('upload_folder')

coref_obj = coref_impl()
# def fast_coref_arch(txt):
    
#     return text


generate_answer = answer_generation()


@app.route('/')
def index():
    return 'Welcome to the PDF Ingestion API!'

@app.route('/uploader', methods = ['PUT'])
def upload_file():
   if request.method == 'PUT':
      file = request.files['file']

   collection = milvus.get_collection()


   if file and file.filename.endswith(".pdf"):

        uploaded_file_path = os.path.join("upload_folder", file.filename)
        print(uploaded_file_path)
        file.save((uploaded_file_path))
        print('staged file locally: ' + str(file.filename))
        
        # Extract text and metadata from the PDF
        text_list, embedding_list, metadata_list = text_processor.extract_text_from_pdf(uploaded_file_path)

        # Insert data into Milvus collection
        print('inseritng into collection')
        #    schema = CollectionSchema(fields=[document_id, metadata, metadata_page, embeddings, text], enable_dynamic_field=True)
        df = pd.DataFrame()
        df['text_list'] = text_list
        df.to_csv('check_text.csv')
    
        print(len(embedding_list))
        print(text_list)
        print(metadata_list)
        collection.insert([ embedding_list, text_list, metadata_list])
        
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

        return jsonify({'message': 'Data inserted into the collection.'}), 200

   return jsonify({'error': 'Invalid file format'}), 400

@app.route('/search-answers', methods=['POST'])
def search_answers():
    data = request.get_json()
    print(data)
    #collection_name = data.get('collection_name', '')
    collection_name = CONF["milvus_connection_name"]
    query = data.get('query', '')
    print(collection_name)
    print(query)
    # Define and load the Milvus collection
    collection = milvus.get_collection()
    collection.load()
    print("Collection loaded.")

    # Encode the query
    clean_query = text_processor.clean_text(query)
    query_encode = text_processor.get_model().encode(clean_query)

    # Perform a search to get answers
    search_results = collection.search(data=[query_encode], anns_field="embeddings",
                                      param={"metric": "L2", "offset": 0},
                                      output_fields=["metadata", "metadata_page", "text"],
                                      limit=10, consistency_level="Strong")
    print(search_results)
    # Extract relevant information from search results
    answers_final = [search_results[0][i].entity.text for i in range(1,len(search_results[0]))]

    #crude reranking
    jaccard_closest_percentage = text_processor.jaccard_sim_list(clean_query, answers_final)
    jaccard_closest = answers_final[jaccard_closest_percentage.index(max(jaccard_closest_percentage))]

    return jsonify({'answers_final': answers_final}), 200

@app.route('/generate-answers', methods=['POST'])
def generate_answers():
    data = request.get_json()
    print(data)
    #collection_name = data.get('collection_name', '')
    collection_name = CONF["milvus_connection_name"]
    query = data.get('query', '')
    print(collection_name)
    print(query)
    # Define and load the Milvus collection
    collection = milvus.get_collection()
    collection.load()
    print("Collection loaded.")

    # Encode the query
    query_encode = text_processor.get_model().encode(query.lower())

    # Perform a search to get answers
    search_results = collection.search(data=[query_encode], anns_field="embeddings",
                                      param={"metric": "L2", "offset": 0},
                                      output_fields=["metadata", "metadata_page", "text"],
                                      limit=10, consistency_level="Strong")
    print(search_results)
    # Extract relevant information from search results
    answers_final = [search_results[0][i].entity.text for i in range(1,len(search_results[0]))]

    generated_ans = generate_answer.openai_answer(query,answers_final)

    return jsonify({'answers_final': generated_ans}), 200

if __name__ == '__main__':
    print("running on 5000 port")
    app.run(debug=False, host='0.0.0.0', port=5000)

