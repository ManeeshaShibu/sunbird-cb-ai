import os
from openai import OpenAI
import json
import requests
class answer_generation:

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        self.client = OpenAI(
        api_key=os.getenv('openai_api_key', self.conf["openai_api_key"]),
        )


    def openai_answer(self, query, context):

        prompt = os.getenv('generative_model_prompt', self.conf["generative_model_prompt"])
        prompt= prompt + context +"}. Now answer my Question which is:" + query
        print(prompt)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            #temperature=1.8,
            model=os.getenv('generative_model', self.conf["generative_model"]),
        )
        return chat_completion.choices[0].message.content

    def mistral_answer(self, query, context):
        mistral_endpoint = os.getenv('mistral_endpoint', self.conf["mistral_endpoint"])
        prompt = os.getenv('generative_model_prompt', self.conf["generative_model_prompt"])
        prompt= prompt + context +"}. Now answer my Question which is:" + query
        print(prompt)
        payload = json.dumps({
            "model": "mistral",
            "prompt": prompt,
            "stream": False
            })
        headers = {
            'Content-Type': 'application/json'
            }
        response = requests.request("POST", mistral_endpoint, headers=headers, data=payload)
        return response.text

        
    def generate_similar_sentences(self, text):
        prompt = os.getenv('generative_model_prompt_simtext', self.conf["generative_model_prompt_simtext"])
        prompt= prompt.replace('placeholder_question' , '{' + text + '}') 
        print(prompt)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=os.getenv('generative_model', self.conf["generative_model"]),
        )
        return chat_completion.choices[0].message.content

