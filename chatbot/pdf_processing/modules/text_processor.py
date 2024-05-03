
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
from modules.pdf_parser import process_pdf
from modules.coref_resolver import coref_impl
from modules.generative_model import answer_generation
from repo.milvus_entity import milvus_collection
import pandas as pd
import json
import assemblyai as aai
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import hashlib
import moviepy.editor as mp
from transformers import pipeline
import whisper

CONF = None
with open('conf/config.json') as config_file:
    CONF = json.load(config_file)


class process_text(coref_impl):
    start_word=""
    def __init__(self, nlp_model, transformer_model) -> None:
        self.nlp = nlp_model
        self.model = transformer_model
        self.coref_obj = coref_impl()        
        self.pdf_processor = process_pdf()
        self.gen_answer = answer_generation()    
        self.buffer_text = []
        pass

    def process(self, text):
        print("in process_text")        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        sentences=[sent for sent in sentences if len(sent)>15 ]### added to remove sentence having length less than 15 characters
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        print("completed_process")
        return sentences, sentence_embeddings

    def cluster_text(self, sentences, sentence_embeddings, threshold):
        clusters = [[0]]
        for i in range(1, len(sentences)):
            similarity = F.cosine_similarity(sentence_embeddings[i - 1:i], sentence_embeddings[i:i + 1]).item()
            if similarity < threshold:
                clusters.append([])
            clusters[-1].append(i)
        return clusters

    def clean_text(self, text):## logic for removing header and footer should be written here
        return self.pdf_processor.text_clean(text)
    
    def doc_signature(self, file):
        name =  os.path.basename(file.name)
        size = os.path.getsize(file.name)
        return name + "_" + str(size)

    def extract_text_from_pdf(self, pdf_path):
        print('pdf processing started')
        text_list = []
        embedding_list = []
        page_list = [] 
        doc_list = [] 
        doc_parent_list = []                          
        with open(pdf_path, 'rb') as pdf_file:
            file_signature = self.doc_signature(pdf_file)            
            file_name=pdf_path.split("\\")[-1].split("__")[-1]
            doc_parent=pdf_path.split("\\")[-1].split("__")[0].split("/")[-1]
            pdf_content = self.pdf_processor.consume_pdf(pdf_path)
            for page in pdf_content['pages']:
                if page["is_outline"]:
                    continue
                text = page['text'].strip()
                if page['header']:
                    text = text.replace(page['header'], ' ')
                if page['footer']:
                    text = text.replace(page['footer'], ' ')
                if len(text.strip())<1:
                    continue
                pagenum = page["page_num"]
                try:
                    text = self.coref_obj.fastcoref_impl(text)
                    if len(text) <=  len(os.getenv('min_chunk_len', CONF["min_chunk_len"])):
                        continue

                    elif len(text) <= 1300:
                        doc_list.append(file_name)
                        text_list.append(text)
                        doc_parent_list.append(doc_parent)
                        embeddings = self.model.encode(text)
                        embedding_list.append(embeddings)
                        page_list.append(str(pagenum))
                    else:
                        txt_list = self.process_large_text(text)
                        for text_chunk in txt_list:
                            embeddings = self.model.encode(text_chunk)
                            embedding_list.append(embeddings)
                            page_list.append(str(pagenum))
                            text_list.append(text_chunk) #### added
                            doc_list.append(file_name)
                            doc_parent_list.append(doc_parent)
                            

                except Exception as e:
                    print("Error:" + str(e))
        
        print(file_name)
        print('pdf processing complete')
        print(len(text_list))
        return text_list, embedding_list, page_list, doc_list, doc_parent_list
    
    def extract_text_from_video(self,f_ile, video_path):
        print('video processing started')
        os.makedirs(".//audio_directory", exist_ok=True)
        audio_path=".//audio_directory//test_1.wav"
        clip = mp.VideoFileClip(video_path, audio_fps=5500)
        clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large")
        out = pipe(audio_path)
        out['text']
        text= out['text']                             
        # aai.settings.api_key = "9bd4c3b823fd42ad9135fb8c8c0b7670"
        # transcriber = aai.Transcriber()
        # transcript = transcriber.transcribe(video_path)        
        text_list = []
        embedding_list = []
        page_list = []
        doc_list = [] 
        doc_parent_list = [] 
        
        print("resolving coref per page")
        file_name=video_path.split("\\")[-1].split("__")[-1]
        doc_parent=video_path.split("\\")[-1].split("__")[0].split("/")[-1]
        def gen():
            n=0
            while True:
                yield n                
                n+=1
        page=gen()
                
        print(len(text))
        pagenum = next(page)
        try:
            text = self.coref_obj.fastcoref_impl(text)

            if len(text) <= 1300:
                print("**************text len in page smaller than 1300")
                print(type(file_name))
                print(file_name)
                text_list.append(text)
                embeddings = self.model.encode(text)
                embedding_list.append(embeddings)
                page_list.append(str(pagenum))
                doc_list.append(file_name)
                doc_parent_list.append(doc_parent)
                
            else:
                print("**************calling large text processor")
                text_list = self.process_large_text(text)
                print("$$$$$$$$$$$" + str(len(text_list)))
                for text_chunk in text_list:
                    embeddings = self.model.encode(text_chunk)
                    embedding_list.append(embeddings)
                    page_list.append(str(pagenum))
                    doc_list.append(file_name)
                    doc_parent_list.append(doc_parent)
                    

        except Exception as e:
            print("Error:" + str(e))
        
        print(file_name)
        print('video processing complete')
        print()
        return text_list, embedding_list, page_list, doc_list, doc_parent_list
    
    def ingest_text(self, query, query_variants, answer):
        text_list = []
        embedding_list = []
        page_list = []

        doc_id = hashlib.shake_128(query.encode('utf-8')).hexdigest(4)

        try:
            page = {"doc_pagenum" : 0, "answer" : answer, "type" : "faq_short", "doc_id" : doc_id, "query" :query, "query_variants" : query_variants}
            text_list.append(query_variants)
            embeddings = self.model.encode(query_variants)
            embedding_list.append(embeddings)
            page_list.append(page)

            #adding one more entry with answer embedded with the question variations 
            page = {"doc_pagenum" : 0, "answer" : answer, "type" : "faq_long", "doc_id" : doc_id,  "query" :query, "query_variants" : query_variants}
            text_list.append(query_variants+ " " + answer)
            embeddings = self.model.encode(query_variants+ " " + answer)
            embedding_list.append(embeddings)
            page_list.append(page)



        except Exception as e:
            print("Error:" + str(e))
        return text_list, embedding_list, page_list

    def process_large_text(self, text):
        print('***************processing large text')
        threshold = 0.3
        sentences, sentence_embeddings = self.process(text)
        clusters = self.cluster_text(sentences, sentence_embeddings, threshold)
        print(clusters, sentences)
        temp_text_list=[]
        for cluster in clusters:

            cluster_txt =' '.join([str(sentences[i]) for i in cluster])
            cluster_len = len(cluster_txt)
            
            # print("*************")
            print(cluster_len)

            if cluster_len < 80:
                print("in if_statement")
                #add to next cluster
                self.buffer_text.append(cluster_txt)
                continue
            elif cluster_len > 1300 and cluster_txt.startswith(self.start_word)== False:
                print("in elif_statement")
                self.cluster_len=cluster_len
                threshold = 0.6
                # print(cluster_txt, text_list)
                self.start_word=""
            
                self.start_word= cluster_txt[:10]
                
                self.process_large_text(cluster_txt)
            else:
                print("in else_statement")
                if self.buffer_text:
                    cluster_txt = ".".join(self.buffer_text) + "." + cluster_txt
                    self.buffer_text = []
                temp_text_list.append(cluster_txt)
        print("going in handle_large_chunks function")
        self.start_word=""
        seperated_text_list = self.handle_lerge_chunks(temp_text_list) ## added new variable name "seperated_text_list" and new "list temp_text_list"
        #for text_chunk in text_list:
        #    page = {"doc_pagenum" : pagenum}
        #    embeddings = self.model.encode(text_chunk)
        #    embedding_list.append(embeddings)
        #    page_list.append(page)
            
        return seperated_text_list## added new variable name

    def jaccard_sim_list(self, source_text, terget_text_list):
        return self.pdf_processor.most_similar_of_list_jaccard(source_text, terget_text_list)

    def handle_lerge_chunks(self, chunks):
        print("handling lerger chunks^^^^^^^^^")
        print(len(chunks))
        chunk_size = 1300
        chunk_overlap = 300
        r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        smaller_chunks = []
        for chunk in chunks:
            if len(chunk)>1300:
                print("big chunk intercepted^^^^^^^^^: " + str(len(chunk)))
                smaller_chunks.extend(r_splitter.split_text(chunk))
            else:
                smaller_chunks.append(chunk)
        print("returning smaller chunks____________")
        print(len(smaller_chunks))
        print(json.dumps(smaller_chunks))
        return smaller_chunks
    
    def generate_multiple_variations(self, query):
        llm_response = self.gen_answer.generate_similar_sentences(query)
        final_text = query
        for sentence in llm_response.split('\n'):
            final_text = final_text + ' ' + sentence.split('. ')[1]
        
        return final_text
    
    
    def get_model(self):
        return self.model