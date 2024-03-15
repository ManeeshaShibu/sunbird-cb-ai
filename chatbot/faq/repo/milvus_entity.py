
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import json
import os
# factor in multiple connections

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
    
    def store_to_milvus(self, sentences_list, embeddings_list, metadata):
        print((metadata))
        print((sentences_list))
        print((embeddings_list))
        self.collection.insert([ embeddings_list, sentences_list, metadata])
        print("embeddings inserted")
        index_params = {
                'metric_type': 'COSINE',
                'index_type': "HNSW",
                'efConstruction': 40,
                'M': 20
            }
        self.collection.create_index(field_name="embeddings", index_params=index_params)
        print('Index created.')

    def search_milvus(self, query_encode):
        res = self.collection.search(data=[query_encode], anns_field="embeddings",
                                        param={"metric": "COSINE", "offset": 0},
                                        output_fields=["metadata", "metadata_page", "text"],
                                        limit=1, consistency_level="Strong")
        return res
    
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

    
 