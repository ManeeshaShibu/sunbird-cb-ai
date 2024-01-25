import PyPDF2 
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument 
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from string_grouper import group_similar_strings
import enchant 
import textdistance as td


import math

import pandas as pd

from pdfminer_util import pdfminer_impl
import io


class process_pdf:

    def __init__(self):
        self.data = []
        self.metadata = {}
        self.outline = None
        self.pagenum = None
        self.total_text = None
        pass

    def outline_checker(self, outline_list, page_text):
        page_text = ' '.join(page_text.split())
        threashhold_percentage = 70
        consecutive_titles = 0
        max_consecutive_titles = 0
        chars_in_consecutive_titles = 0
        for title in outline_list:
            
            if ' '.join(title.split()) in page_text:
                consecutive_titles = consecutive_titles+1
                if max_consecutive_titles < consecutive_titles:
                    max_consecutive_titles = consecutive_titles
                    chars_in_consecutive_titles = chars_in_consecutive_titles + len(title)
            elif max_consecutive_titles <= 1 :
                consecutive_titles = 0
                chars_in_consecutive_titles = 0
        count_alnum_chars = sum(c.isdigit() for c in page_text) + sum(c.isalpha() for c in page_text)
      
        if chars_in_consecutive_titles> 0 and (chars_in_consecutive_titles/count_alnum_chars)*100 > threashhold_percentage:
            print("==========")
            print(page_text)
            print(max_consecutive_titles)
            print(chars_in_consecutive_titles)
            print("==========")
            return True
        return False

    def list_pdf_lines(self, pdf_file):
        #print("================")
        text_list = []
        for page_layout in extract_pages(pdf_file):
            #print("================")
            text_set = set()
            for element in page_layout:
                
                if isinstance(element, LTTextContainer) and element.get_text() and not element.get_text().isspace():
                    #print("----------------------------")
                    text_set.add(element.get_text())
            text_list.extend(list(text_set))
        self.total_text =  text_list

    def possible_header_footer(self):
        pagenum_thresh_percent = 90
        pdf_min_pages_with_footer_header = math.floor((self.pagenum*pagenum_thresh_percent)/100)
        strings = pd.DataFrame()
        strings['str'] = self.total_text


        df = group_similar_strings(
                strings['str'], 
                min_similarity=0.6)

        df['frequency'] = df.groupby(["group_rep_index"]).transform("count")
        df = df[df['frequency'].between(pdf_min_pages_with_footer_header, self.pagenum)]
        return df['group_rep_str'].unique().tolist()



    def get_text_lines(self, text):
        lines = text.splitlines()
        non_empty_alphanumeric_lines = []
        for line in lines:
            if line.strip():
               non_empty_alphanumeric_lines.append(line) 
        return non_empty_alphanumeric_lines
    

    
    def get_outline(self, pdf_file):
        #check if pdf page is outlie page
        parser = PDFParser(pdf_file)
        document = PDFDocument(parser)
        try:
            outlines = document.get_outlines()
            self.outline = outlines
            outline_list = []
            for(level,title,dest,a,se) in outlines:
                outline_list.append(title)
            print("has outline")
            return True, outline_list
        except Exception as e:
            pass
        return False, None
        
    def pdf_to_text(self, pdf_file):
        #extract text from pdf file and return
        pdf_to_txt = pdfminer_impl()
        return pdf_to_txt.extract_text_without_images_and_tables(pdf_file)
    
    def get_if_header(self, content):
        lines = self.get_text_lines(content)
        similarity_perc = [td.jaccard.normalized_similarity(lines[0], s) for s in self.possible_header_footer()]

        if len(lines)>0 and len(similarity_perc)>0 and max(similarity_perc) > 0.8:
            return lines[0]
        else:
            return None
        
    def get_if_footer(self, content):
        lines = self.get_text_lines(content)
        similarity_perc = [td.jaccard.normalized_similarity(lines[-1], s) for s in self.possible_header_footer()]
        
        if len(lines)>0 and len(similarity_perc)>0 and max(similarity_perc) > 0.8:
            return lines[-1]
        else:
            return None
        
    def process_pages(self, pdf_file):
        #break pdf file into multiple pdfs, single page in a pdf
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages = []
        has_outline, outline = self.get_outline(pdf_file)
        self.pagenum = len(pdf_reader.pages)
        for page_number in range(len(pdf_reader.pages)):
            content = {}
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_number])

            in_memory_pdf_file = io.BytesIO()
            pdf_writer.write(in_memory_pdf_file)
            in_memory_pdf_file.seek(0)
            with open('data/tmp/' + str(page_number) + '.pdf',"wb") as tmp_file:
                tmp_file.write(in_memory_pdf_file.read())
            content['page_num'] = page_number
            content['text'] = self.pdf_to_text(in_memory_pdf_file)
            content['header'] = self.get_if_header(content['text'])
            content['footer'] = self.get_if_footer(content['text'])
            if has_outline:
                content['is_outline'] = self.outline_checker(outline, content['text'])
            else:
                content['is_outline'] = False
            pages.append(content)
        
        self.data = pages

    def consume_pdf(self, file_name):
        #take the pdf and return text and metadata
        fp = open(file_name, 'rb')
        self.list_pdf_lines(fp)
        self.process_pages(fp)
        return {'pages' : self.data}
