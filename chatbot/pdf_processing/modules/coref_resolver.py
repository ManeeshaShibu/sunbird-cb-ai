import spacy
import re
from fastcoref import spacy_component

class coref_impl:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe(
            "fastcoref",
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
        )
        print("fastcoref added to nlp pipeline")

    def fastcoref_impl(self, text): 
        print("coreferencing using fastcoref")    
        doc = self.nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
        text = doc._.resolved_text
        text = re.sub("\n", " ", text)
        text = text.lower()
        return text
    

print(coref_impl().fastcoref_impl("jack and jill went up the hill. jack started crying. he was sad"))
