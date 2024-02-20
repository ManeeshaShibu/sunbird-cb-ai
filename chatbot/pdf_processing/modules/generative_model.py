import os
from openai import OpenAI
import json

class answer_generation:

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        self.client = OpenAI(
        api_key=self.conf['openai_api_key'],
        )


    def openai_answer(self, query, context):
        context_count = len(context)
        if self.conf['top_matching_chunks_as_context'] < context_count:
             context_count = self.conf['top_matching_chunks_as_context']
        prompt = self.conf['generative_model_prompt']
        prompt= prompt + ' '.join(context[:context_count]) +"}. Now answer my Question which is:" + query
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
        return chat_completion.choices[0].message.content

        
    