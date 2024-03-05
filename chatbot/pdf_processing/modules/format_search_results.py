

class format_answer:

    def __init__(self) -> None:
        pass

    def return_combined_chunks(self, main_chunks, priority_chunks, top_n):
        combined_list = main_chunks
        for chunk in priority_chunks:
            print(chunk)
            print({'similarity_distacne' : chunk["similarity_distacne"], 'text-chunk': chunk["text-chunk"]["answer"]})
            combined_list.append({'similarity_distacne' : chunk["similarity_distacne"], 'text-chunk': chunk["text-chunk"]["answer"]})

        new_list = sorted(combined_list, key=lambda x: x["similarity_distacne"], reverse=False)
        print("===================")
        print(new_list)
        print("===================")
        
        
        return new_list[:int(top_n)]