
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import json



class milvus_collection:

    collection = None
    conf = None
    collection_loaded = False

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        connections.connect(host=self.conf["milvus_host"], port=self.conf["milvus_port"])
        self.define_collection(self.conf["milvus_connection_name"])
        print('Connected to Milvus!')

    def define_collection(self, collection_name):
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
        
        self.collection = collection
        return self.collection
    
    def get_collection(self):
        return self.collection
    
    def load_collection(self):
        if not self.collection_loaded:
            self.collection.load()
            self.collection_loaded = True
        return self.collection
    
    def drop_collection(self, collection_name):
        utility.drop_collection(collection_name)

    
 