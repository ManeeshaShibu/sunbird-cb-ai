
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import json
import os
# factor in multiple connections
from ctransformers import AutoModelForCausalLM,AutoConfig

config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF")
# Explicitly set the max_seq_len
config.config.max_new_tokens = 2048
config.config.context_length = 4096
config.config.temperature=0.70
class milvus_collection:

    collection = None
    conf = None

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        connections.connect(host=os.getenv('milvus_host', self.conf["milvus_host"]), port=os.getenv('milvus_port', self.conf["milvus_port"]))
        self.define_collection(os.getenv('milvus_collection_name', self.conf["milvus_collection_name"]))
        print('Connected to Milvus!')

    def define_collection(self, collection_name):
        document_id = FieldSchema(name='document_id', dtype=DataType.INT64, is_primary=True, auto_id=True)
        text = FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=60000)
        embeddings = FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=384)
        page = FieldSchema(name='page', dtype=DataType.VARCHAR, max_length=100)
        doc = FieldSchema(name='doc_name', dtype=DataType.VARCHAR, max_length=100)
        doc_parent = FieldSchema(name='doc_parent', dtype=DataType.VARCHAR, max_length=100)       
        
        
        schema = CollectionSchema(fields=[document_id, embeddings, text, page, doc, doc_parent], enable_dynamic_field=True)
        
        if not utility.has_collection(collection_name):
            collection = Collection(name=collection_name, schema=schema, using='default')
            print('Collection created!')
        else:
            collection = Collection(collection_name)
            print('Collection already exists.')
        
        self.collection = collection
        return self.collection
    
    def get_collection(self):
        return self.collection
    
    def load_collection(self):
        self.collection.load()
        return self.collection
    
    def drop_collection(self, collection_name):
        utility.drop_collection(collection_name)
        return "dropped " + collection_name
    
    def get_collections(self):
        return utility.list_collections()

    
 