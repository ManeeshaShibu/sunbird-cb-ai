
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
from repo.milvus_entity import milvus_collection
import pandas as pd
import json
import assemblyai as aai
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


class process_text:
    cluster_len=0
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.coref_obj = coref_impl()
        self.pdf_processor = process_pdf()
        pass

    def process(self, text):
        doc = self.nlp(text)
        sentences = list(doc.sents)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
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
                             
        with open(pdf_path, 'rb') as pdf_file:
            file_signature = self.doc_signature(pdf_file)
            print(pdf_path)
            
            pdf_content = self.pdf_processor.consume_pdf(pdf_path)
            text_list = []
            embedding_list = []
            metadata_list = []
            print("resolving coref per page")
            for page in pdf_content['pages']:
                #cleanup
                if page["is_outline"]:
                    #skip the TOC pages
                    continue
                text = page['text'].strip()
                if page['header']:
                    text = text.replace(page['header'], ' ')
                if page['footer']:
                    text = text.replace(page['footer'], ' ')
                if len(text.strip())<1:
                    continue

                print(page["page_num"])
                print(len(text))
                pagenum = page["page_num"]
                try:
                    text = self.coref_obj.fastcoref_impl(text)
                    # print(text)

                    if len(text) <= 1300:
                        print("**************text len in page smaller than 1300")
                        
                        metadata = {"doc_pagenum" : pagenum, "doc_name" : os.path.basename(pdf_file.name), "doc_signature" : file_signature}
                        text_list.append(text)
                        # print(text)
                        embeddings = self.model.encode(text)
                        embedding_list.append(embeddings)
                        metadata_list.append(metadata)
                    else:
                        print("**************calling large text processor")
                        text_list = self.process_large_text(text, pdf_path, pagenum, text_list, embedding_list, metadata_list)
                        for text_chunk in text_list:
                            metadata = {"doc_pagenum" : pagenum}
                            embeddings = self.model.encode(text_chunk)
                            embedding_list.append(embeddings)
                            metadata_list.append(metadata)


                except Exception as e:
                    print("Error:" + str(e))
        

        print('pdf processing complete')
        print()
        return text_list, embedding_list, metadata_list
    
    def extract_text_from_video(self,file_name, video_path):
        print('video processing started')
                             
        aai.settings.api_key = "9bd4c3b823fd42ad9135fb8c8c0b7670"
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(video_path)
        
        text_list = []
        embedding_list = []
        metadata_list = []
        print("resolving coref per page")
        
        text = transcript.text.strip()
        print(text)
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
            # print(text)

            if len(text) <= 1300:
                print("**************text len in page smaller than 1300")
                
                metadata = {"doc_pagenum" : pagenum, "doc_name" : os.path.basename(file_name.name)}
                text_list.append(text)
                # print(text)
                embeddings = self.model.encode(text)
                embedding_list.append(embeddings)
                metadata_list.append(metadata)
            else:
                print("**************calling large text processor")
                text_list = self.process_large_text(text, video_path, pagenum, text_list, embedding_list, metadata_list)
                print("$$$$$$$$$$$" + str(len(text_list)))
                for text_chunk in text_list:
                    metadata = {"doc_pagenum" : pagenum}
                    embeddings = self.model.encode(text_chunk)
                    embedding_list.append(embeddings)
                    metadata_list.append(metadata)

        except Exception as e:
            print("Error:" + str(e))
        

        print('video processing complete')
        print()
        return text_list, embedding_list, metadata_list

    def process_large_text(self, text, pdf_path, pagenum, text_list, embedding_list, metadata_list):
        print('***************processing large text')
        threshold = 0.3
        sentences, sentence_embeddings = self.process(text)
        clusters = self.cluster_text(sentences, sentence_embeddings, threshold)

        for cluster in clusters:

            cluster_txt =' '.join([str(sentences[i]) for i in cluster])
            cluster_len = len(cluster_txt)
            
            # print("*************")
            # print(cluster_len)

            if cluster_len < 80:
                #add to next cluster
                continue
            elif cluster_len > 1300 and self.cluster_len!=cluster_len:
                self.cluster_len=cluster_len
                threshold = 0.6
                # print(cluster_txt, text_list)
                
                self.process_large_text(cluster_txt, pdf_path, pagenum, text_list, embedding_list, metadata_list)
            else:
                text_list.append(cluster_txt)
        text_list = self.handle_lerge_chunks(text_list)
        #for text_chunk in text_list:
        #    metadata = {"doc_pagenum" : pagenum}
        #    embeddings = self.model.encode(text_chunk)
        #    embedding_list.append(embeddings)
        #    metadata_list.append(metadata)
            
        self.cluster_len=0
        return text_list

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

    def get_model(self):
        return self.model