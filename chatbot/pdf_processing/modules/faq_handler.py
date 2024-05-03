import pandas as pd
import os
import json
import requests


class faq:

    def __init__(self) -> None:
        with open("conf/config.json") as config_file:
            self.conf = json.load(config_file)
        self.faq_url_ingest = os.getenv("faq_url_ingest", self.conf["faq_url_ingest"])
        self.faq_url_query = os.getenv("faq_url_query", self.conf["faq_url_query"])

    def load_faq(self, faq_json):
        payload = json.dumps({"faqs": faq_json})
        headers = {"Content-Type": "application/json"}

        response = requests.request(
            "POST", self.faq_url_ingest, headers=headers, data=payload
        )
        print(response.text)

    def query(self, ques):
        payload = json.dumps({"query": ques})
        headers = {"Content-Type": "application/json"}

        response = requests.request(
            "POST", self.faq_url_query, headers=headers, data=payload
        )

        if response.status_code == 200:
            print(response.text)
            resp_obj = json.loads(response.text)
            return resp_obj, resp_obj['score']

        else:
            return False, False
