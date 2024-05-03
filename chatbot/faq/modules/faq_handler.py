import pandas as pd
import os
import json
from sentence_transformers import SentenceTransformer, util
from repo.milvus_entity import milvus_collection

class faq:
    
    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        self.ques_list = []
        self.ans_list = []
        self.milvus = milvus_collection()
        self.model = SentenceTransformer(os.getenv('encoding_model', self.conf['encoding_model']))

    def load_faq(self, faq_json):
        faq_data = pd.DataFrame()
        print(type(faq_json))
        for obj in faq_json:
            print(obj)
            self.ques_list.append(obj['query'])
            self.ans_list.append({'answer' : obj['answer']})

        faq_data['question'] = self.ques_list
        faq_data["question"] = faq_data["question"].apply(lambda x: x.strip())
        
        self.ques_list = list(faq_data["question"].values)
        embeddings = self.gen_emb(self.ques_list)
        self.milvus.store_to_milvus(self.ques_list, embeddings, self.ans_list)
        self.milvus.load_collection()

        

    def gen_emb(self, sentences_list):
        #sentences_list = get_queries_wordlist()
        
        embeddings_list = self.model.encode(sentences_list, convert_to_tensor=False, normalize_embeddings=True)
        return embeddings_list
    
    def query(self, ques):
        self.milvus.load_collection()
        ques_emb = self.gen_emb([ques])
        res = self.milvus.search_milvus(ques_emb[0])
        answer = []
        for result in res:
            for r in result:
                print(r)
                answer = {"text-chunk" : r.entity.text, "similarity_distacne" : r.distance, "answer" : r.entity.metadata}
        return answer, answer['similarity_distacne']
    

