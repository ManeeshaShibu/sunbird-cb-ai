
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import json
import os
# factor in multiple connections

class milvus_collection:

    collection = None
    priority_collection = None
    conf = None

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        connections.connect(host=os.getenv('milvus_host', self.conf["milvus_host"]), port=os.getenv('milvus_port', self.conf["milvus_port"]))
        self.define_collection(os.getenv('milvus_collection_name', self.conf["milvus_collection_name"]), os.getenv('milvus_priority_collection_name', self.conf["milvus_priority_collection_name"]))
        print('Connected to Milvus!')

    def define_collection(self, collection_name, priority_collection_name):
        document_id = FieldSchema(name='document_id', dtype=DataType.INT64, is_primary=True, auto_id=True)
        metadata = FieldSchema(name='metadata', dtype=DataType.JSON, max_length=15000)
        embeddings = FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=384)
        text = FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=60000)
        schema = CollectionSchema(fields=[document_id, embeddings, text, metadata], enable_dynamic_field=True)
        
        if not utility.has_collection(collection_name):
            collection = Collection(name=collection_name, schema=schema, using='default')
            print('Collection created!')
        else:
            collection = Collection(collection_name)
            print('Collection already exists.')
        
        if not utility.has_collection(priority_collection_name):
            priority_collection = Collection(name=priority_collection_name, schema=schema, using='default')
            print('Priority Collection created!')
        else:
            priority_collection = Collection(priority_collection_name)
            print('Priority Collection already exists.')
        
        self.collection = collection
        self.priority_collection = priority_collection
        return self.collection, self.priority_collection
    
    def get_collection(self):
        return self.collection, self.priority_collection
    
    def load_collection(self):
        self.collection.load()
        return self.collection
    
    def drop_collection(self, collection_name):
        utility.drop_collection(collection_name)
        return "dropped " + collection_name
    
    def get_collections(self):
        return utility.list_collections()

    
 