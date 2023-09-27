from flask import Flask, request, jsonify
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/get": {"origins": "http://localhost:3000"}})

# Download tokenization data and stopwords data from the server
nltk.downloader.download('punkt')
nltk.downloader.download('stopwords')

# Function to fetch data from the API and store it globally
global_var = None

def get_data_from_api():
    global global_var
    api_url = 'https://backend69.up.railway.app/get/projects'

    # Make the GET request
    response = requests.get(api_url)
    if response.status_code == 200:
        global_var = response.json()
    else:
        print("Error: Failed to retrieve data from the API")
        
    global_var = global_var.get('message')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
   
# Query parsing and synonym handling
def parse_query(query):
    # Tokenize the query
    tokens = nltk.word_tokenize(query)
    
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords]
    
    # Lemmatize tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Synonym handling using WordNet
    synonyms = []
    for word in lemmatized_tokens:
        synsets = wordnet.synsets(word)
        synonyms.extend([synset.lemmas()[0].name() for synset in synsets])
        
        synonyms.append(word)

    # Convert the list of synonyms into a single string
    query_str = " ".join(synonyms)
    print(query_str)
    
        
    return query_str

def parse_projectdata(projectdata):
    # Tokenize the query
    tokens = nltk.word_tokenize(projectdata)
    preprocessed_descriptions = []
    
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords]
    
    # Lemmatize tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Lowercase each term and join tokens back into a single string
    preprocessed_description = ' '.join([token.lower() for token in lemmatized_tokens])
    preprocessed_descriptions.append(preprocessed_description)

    return preprocessed_descriptions


# API endpoint to check plagiarism

@app.route('/search', methods=['POST'])
@cross_origin(origin="http://localhost:3000", headers=["Content-Type", "Authorization"])
def api_check_plagiarism():
    try:
        global global_var
        if global_var is None:
            get_data_from_api()
        
        data = request.get_json()
        user_query = data['query']
        
        preprocessed_user_query = parse_query(user_query)
        
        cosine_similarities = []  # Create an empty list to store cosine similarities
        
        for item in global_var:
            if 'title' in item:
                title = item['title']
                
            if 'shortdiscription' in item:
                short_description = item['shortdiscription']
                
            if 'description' in item:
                description = item['description']
                
                
            project_data = title + " " + short_description + " " + description
                
            preprocessed_project_data = parse_projectdata(project_data)
            
            tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_user_query] + preprocessed_project_data)
            cosine_similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # Append the cosine similarity score for this project
            cosine_similarities.append(cosine_similarity_scores[0][0])
        
        # Rank search results based on cosine similarity scores
        ranked_results = [(id['_id'], cosine_similarity_score) for id, cosine_similarity_score in zip(global_var, cosine_similarities)]
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Return the ranked results as JSON
        response = [{"_id": id, "cosine_similarity_score": score} for id, score in ranked_results]

        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)