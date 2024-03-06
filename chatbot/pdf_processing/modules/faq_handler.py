from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
import pandas as pd
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
import os
import json

class faq:
    
    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        self.document_store = InMemoryDocumentStore()
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=True,
            scale_score=False,
        )
        self.ques_list = []
        self.ans_list = []
        
        self.pipe = []

    def load_faq(self, faq_json):
        faq_data = pd.DataFrame()
        for obj in faq_json:
            self.ques_list.append(obj['heading'])
            self.ans_list.append(obj['value'])
        faq_data['question'] = self.ques_list
        faq_data["question"] = faq_data["question"].apply(lambda x: x.strip())
        faq_data['answer'] = self.ans_list

        questions = list(faq_data["question"].values)
        faq_data["embedding"] = self.retriever.embed_queries(queries=questions).tolist()
        faq_data = faq_data.rename(columns={"question": "content"})
        docs_to_index = faq_data.to_dict(orient="records")
        self.document_store.write_documents(docs_to_index)

        self.pipe = FAQPipeline(retriever=self.retriever)

    def query(self, ques):
        prediction = self.pipe.run(query=ques, params={"Retriever": {"top_k": 1}})
        #print_answers(prediction, details="medium")
        answer =  prediction['answers'][0]
        if answer.score > float(os.getenv('haystack_faq_cutoff', self.conf["haystack_faq_cutoff"])):
            return answer.answer, answer.score
        else:
            return False, False
    

