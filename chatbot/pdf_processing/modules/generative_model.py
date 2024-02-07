import os
from openai import OpenAI
import json

class answer_generation:

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        self.client = OpenAI(
        api_key='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',
        )


    def openai_answer(self, query, context):
        context_count = len(context) 
        if context_count > self.conf["top_matching_chunks_as_context"]:
            context_count = self.conf["top_matching_chunks_as_context"]
        prompt= self.conf["generative_model_prompt"] + ' '.join(context[:context_count]) +". My Question is:" + query
        print(prompt)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        print(chat_completion)
        if not chat_completion.choices==[]:
            return "We couldn't find any answer for the question"
        
        return chat_completion.choices[0].message.content
    
    