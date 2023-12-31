# procollab-search-engine-backend
Explaining the search engine Backend of ProCollab:

Hi, everyone! I'm here to explain the backend I've built for ProCollab, a project collaboration platform.

1. Building the Backend with Flask:
We start by using Flask, a Python framework, to create our backend server. Think of Flask as the engine that powers our application.

2. Handling Cross-Origin Requests:
To make things easy for our web app, we use Flask-CORS. This allows our frontend, to talk to our backend.

3. Data Preparation with NLTK:
For text processing, we turn to NLTK, a natural language processing library. We download data like tokenization rules and stopwords from NLTK.

4. Global Data Storage:
To store the project data that we fetch from an external API, we use a global variable called global_var. It's like our digital whiteboard for project details.

5. Fetching Data from an API:
We have a function called get_data_from_api. It makes an API call to an external source that holds our project information. The fetched data goes into global_var.

6. TF-IDF Vectorization:
We're using something called TF-IDF (Term Frequency-Inverse Document Frequency) to understand text better. It helps us convert words into numbers, which is easier for a computer to work with.

7. User Query Processing:
When a user types a search query, we tokenize it (split it into words), remove common words (stopwords), and reduce words to their base form (lemmatization). We also expand the query with synonyms to capture more possibilities.

8. Project Data Processing:
Similarly, we process the project data from our API using the same steps as with the user query. This levels the playing field for comparison.

9. Checking for Plagiarism:
Now, the magic happens! We calculate something called "cosine similarity" between the user's query and each project's data. This tells us how much they match. We store these similarity scores in a list.

10. Ranking the Results:
To give users the best results, we sort the projects based on their similarity scores. The ones with higher scores come up first.

11. Responding to the Frontend:
Finally, we send back the ranked results to our web app. Each result includes the project ID and its similarity score.

12. Running the Server:
We run our Flask app with app.run(), making it available for our web app to communicate with.