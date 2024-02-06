import os
os.environ["OPENAI_API_KEY"] = "sk-abc123abc123abc123abc123abc123abc123abc123abc123"
import json
from openai import OpenAI

import numpy as np

class answer_generation:

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        print(self.conf["openai_api_key"])
        OpenAI.api_key = self.conf["openai_api_key"]
        self.client = OpenAI()


    def openai_answer(self, query, context):
        context_count = len(context) 
        if context_count > self.conf["top_matching_chunks_as_context"]:
            context_count = self.conf["top_matching_chunks_as_context"]
        prompt= self.conf["generative_model_prompt"] + ' '.join(context[:context_count]) +". My Question is:" + query
        print(prompt)
        prompt_answer_response = self.client.completions.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            temperature=0,
            max_tokens=200,
        )
        if(prompt_answer_response['choices']==[]):
            return "We couldn't find any answer for the question"
        
        return prompt_answer_response['choices'][0]['text']
