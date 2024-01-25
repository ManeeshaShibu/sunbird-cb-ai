from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class sim:

    def __init__(self):
        pass

    def most_similar(self, query, documents):


        # Combine the query with the documents
        all_texts = [query] + documents

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit and transform the documents into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Calculate cosine similarity between the query and all documents
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        # Find the most similar document to the query
        most_similar_index = cosine_similarities.argmax()
        most_similar_doc = documents[most_similar_index]
        similarity_score = cosine_similarities[most_similar_index]

        print(f"Query: {query}")
        print(f"Most Similar Document: {most_similar_doc}")
        print(f"Similarity Score: {similarity_score:.4f}")


documents = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]
query = "This is a new document number two."

smi = sim()

smi.most_similar(query, documents)