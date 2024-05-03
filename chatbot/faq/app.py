from flask import Flask, request, jsonify
from modules.faq_handler import faq


app = Flask(__name__)
faq_obj = faq()


@app.route('/ingest/faq', methods = ['POST'])
def ingest_content():
    data = request.get_json()
    faqs =  data['faqs']
    faq_obj.load_faq(faqs)
    return jsonify({'message': 'loaded the FAQs'}), 200

@app.route('/faq/query', methods=['POST'])
def generate_answers():
    data = request.get_json()
    print(data)
    #collection_name = data.get('collection_name', '')
    query = data.get('query', '')
    faq_response, score = faq_obj.query(query)
    if faq_response:
        print({'generated_ans': faq_response, 'closest context' : "faq", "score" : score})
        return jsonify({'generated_ans': faq_response, 'closest context' : "faq", "score" : score}), 200 
    else:
        return jsonify({'generated_ans': '', 'closest context' : "faq", "score" : 0}), 404 
    

if __name__ == '__main__':
    print("running on 5000 port")
    app.run(debug=False, host='0.0.0.0', port=5001)