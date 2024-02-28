import os
from openai import OpenAI
import json

class answer_generation:

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        self.client = OpenAI(
        api_key=os.getenv('openai_api_key', self.conf["openai_api_key"]),
        )


    def openai_answer(self, query, context):
        context_count = len(context)
        top_context_chunk_count = os.getenv('top_matching_chunks_as_context', self.conf["top_matching_chunks_as_context"])
        if top_context_chunk_count < context_count:
             context_count = top_context_chunk_count
        prompt = os.getenv('generative_model_prompt', self.conf["generative_model_prompt"])
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

        
    