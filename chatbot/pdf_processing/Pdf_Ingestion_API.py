#### Pdf ingestion API
# http://127.0.0.1:5000//upload-pdf/<collection_name>
# payload format body-form_data-key(file(default))- value(.pdf file path)


from flask import Flask, request, jsonify
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import os
import spacy
import re
import PyPDF2
from fastcoref import spacy_component
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from werkzeug.utils import secure_filename
from fastcoref import spacy_component
from pdf_parser import process_pdf

app = Flask(__name__)

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
print('Connected to Milvus!')

# Create 'upload_folder' directory if it doesn't exist
if not os.path.exists('upload_folder'):
    os.makedirs('upload_folder')

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

# def fast_coref_arch(txt):
    
#     return text

pdf_processor = process_pdf()

def define_collection(collection_name):
    document_id = FieldSchema(name='document_id', dtype=DataType.INT64, is_primary=True, auto_id=True)
    metadata = FieldSchema(name='metadata', dtype=DataType.VARCHAR, max_length=15000)
    metadata_page = FieldSchema(name='metadata_page', dtype=DataType.INT64)    
    embeddings = FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=384)
    text = FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=60000)
    schema = CollectionSchema(fields=[document_id, metadata, metadata_page, embeddings, text], enable_dynamic_field=True)
    
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema, using='default')
        print('Collection created!')
    else:
        collection = Collection(collection_name)
        print('Collection already exists.')
    return collection

def process(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    return sentences, sentence_embeddings

def cluster_text(sentences, sentence_embeddings, threshold):
    clusters = [[0]]
    for i in range(1, len(sentences)):
        similarity = F.cosine_similarity(sentence_embeddings[i - 1:i], sentence_embeddings[i:i + 1]).item()
        if similarity < threshold:
            clusters.append([])
        clusters[-1].append(i)
    return clusters

def clean_text(text):## logic for removing header and footer should be written here
    return text

def extract_text_from_pdf(pdf_path):
    print('pdf processing started')
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "fastcoref",
        config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
    )                
    print("fastcoref pipeline loaded")
    with open(pdf_path, 'rb') as pdf_file:
        print(pdf_path)
        pdf_content = pdf_processor.consume_pdf(pdf_path)
        text_list = []
        embedding_list = []
        metadata_list = []
        metadata_page_list = []
        print("resolving coref per page")
        for page in pdf_content['pages']:
            
            text = page['text']
            print(text)
            # print(len(text))
            try:
                doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
                text = doc._.resolved_text
                text = re.sub("\n", " ", text)
                text = text.lower()
                # print(text)

                if len(text) <= 1300:
                    metadata = f"{pdf_path}_{page}"
                    metadata_page = page
                    text_list.append(text)
                    # print(text)
                    embeddings = model.encode(text)
                    embedding_list.append(embeddings)
                    metadata_list.append(metadata)
                    metadata_page_list.append(metadata_page)
                else:
                    print("calling large text processor")
                    process_large_text(text, pdf_path, page, text_list, embedding_list, metadata_list, metadata_page_list)

            except:
                print("error")
                pass
        print(text_list)
    print('pdf processing complete')
    
    return text_list, embedding_list, metadata_list, metadata_page_list

def process_large_text(text, pdf_path, page, text_list, embedding_list, metadata_list, metadata_page_list):
    print('processing large text')
    threshold = 0.3
    sentences, sentence_embeddings = process(text)
    clusters = cluster_text(sentences, sentence_embeddings, threshold)

    for cluster in clusters:
        cluster_txt = clean_text(' '.join([str(sentences[i]) for i in cluster]))
        cluster_len = len(cluster_txt)
        # print("*************")
        # print(cluster_len)

        if cluster_len < 80:
            continue
        elif cluster_len > 1300:
            threshold = 0.6
        #    process_large_text(cluster_txt, pdf_path, page, text_list, embedding_list, metadata_list, metadata_page_list)
        else:
            metadata = f"{pdf_path}_{page}"
            metadata_page = page
            text_list.append(cluster_txt)
            embeddings = model.encode(cluster_txt)
            embedding_list.append(embeddings)
            metadata_list.append(metadata)
            metadata_page_list.append(metadata_page)
        # print(text_list)

@app.route('/')
def index():
    return 'Welcome to the PDF Ingestion API!'

@app.route('/uploader', methods = ['PUT'])
def upload_file():
   if request.method == 'PUT':
      file = request.files['file']

   collection_name = 'default'
   collection = define_collection(collection_name)


   if file and file.filename.endswith(".pdf"):

        uploaded_file_path = os.path.join("upload_folder", file.filename)
        print(uploaded_file_path)
        file.save((uploaded_file_path))
        print('staged file locally: ' + str(file.filename))
        
        # Extract text and metadata from the PDF
        text_list, embedding_list, metadata_list, metadata_page_list = extract_text_from_pdf(uploaded_file_path)

        # Insert data into Milvus collection
        print('inseritng into collection')
        collection.insert([metadata_list, metadata_page_list, embedding_list, text_list])

        # Create an index on the "embeddings" field
        index_params = {
            'metric_type': 'L2',
            'index_type': "HNSW",
            'efConstruction': 40,
            'M': 20
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        print('Index created.')

        return jsonify({'message': 'Data inserted into the collection.'}), 200

   return jsonify({'error': 'Invalid file format'}), 400

@app.route('/search-answers', methods=['POST'])
def search_answers():
    data = request.get_json()
    print(data)
    #collection_name = data.get('collection_name', '')
    collection_name = 'default'
    query = data.get('query', '')
    print(collection_name)
    print(query)
    # Define and load the Milvus collection
    collection = define_collection(collection_name)
    collection.load()
    print("Collection loaded.")

    # Encode the query
    query_encode = model.encode(query.lower())

    # Perform a search to get answers
    search_results = collection.search(data=[query_encode], anns_field="embeddings",
                                      param={"metric": "L2", "offset": 0},
                                      output_fields=["metadata", "metadata_page", "text"],
                                      limit=10, consistency_level="Strong")
    print(search_results)
    # Extract relevant information from search results
    answers_final = [search_results[0][i].entity.text for i in range(1,len(search_results[0]))]

    return jsonify({'answers_final': answers_final}), 200

if __name__ == '__main__':
    print("running on 5000 port")
    app.run(debug=False, host='0.0.0.0', port=5000)

