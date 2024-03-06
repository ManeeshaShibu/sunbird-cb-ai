import json
import os
class format_answer:

    def __init__(self) -> None:
        with open('conf/config.json') as config_file:
            self.conf = json.load(config_file)
        

    def return_combined_chunks(self, main_chunks, priority_chunks, top_n):
        combined_list = main_chunks
        absolute_answer = False
        priority_list = []


        for chunk in priority_chunks:
            print(chunk)
            print("===============&&&&&")
            print({'similarity_distacne' : chunk["similarity_distacne"], 'text-chunk': chunk["text-chunk"]["answer"]})
            query = chunk["text-chunk"]["query"]
            combined_list.append({'similarity_distacne' : chunk["similarity_distacne"], 'text-chunk': query  + " " + chunk["text-chunk"]["answer"]})
            priority_list.append({'similarity_distacne' : chunk["similarity_distacne"], 'text-chunk': chunk["text-chunk"]["answer"]})

        new_list = sorted(combined_list, key=lambda x: x["similarity_distacne"], reverse=False)
        sorted_priority_list = sorted(priority_list, key=lambda x: x["similarity_distacne"], reverse=False)
        if sorted_priority_list[0]["similarity_distacne"] < int(os.getenv('similarity_distance_cutoff_faq', self.conf["similarity_distance_cutoff_faq"])):
            absolute_answer = True
            return sorted_priority_list[0], absolute_answer
        print("===================")
        print(new_list)
        print("===================")
        
        
        return new_list[:int(top_n)], absolute_answer