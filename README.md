# ProCollab Search Engine Backend

The **ProCollab Search Engine Backend** powers the search functionality for **ProCollab**, a project collaboration platform. This backend processes user queries and project data, delivering ranked results using natural language processing (NLP) techniques.

---

## Features

1. **Flask-Based Backend**:
   - Built using Flask, a lightweight Python framework, to handle server-side logic.

2. **Cross-Origin Requests**:
   - Uses `Flask-CORS` to enable seamless communication between the frontend and backend.

3. **Data Processing with NLTK**:
   - Leverages the **Natural Language Toolkit (NLTK)** for tokenization, stopword removal, and lemmatization.

4. **API Integration**:
   - Fetches project data from an external API and stores it in a global variable (`global_var`).

5. **TF-IDF Vectorization**:
   - Transforms text into numerical representations for similarity calculations.

6. **Query Expansion**:
   - Enhances user search queries by adding synonyms to improve search results.

7. **Cosine Similarity**:
   - Calculates the similarity between user queries and project data to rank results.

8. **Result Ranking**:
   - Returns projects ranked by their similarity scores for the best user experience.

---

## How It Works

### 1. Fetching and Preparing Data:
- Project data is fetched from an external API using the `get_data_from_api()` function.
- Both user queries and project data undergo text preprocessing:
  - Tokenization
  - Stopword removal
  - Lemmatization

### 2. Processing User Queries:
- User queries are expanded with synonyms to improve search coverage.
- Preprocessed queries are transformed into TF-IDF vectors.

### 3. Calculating Similarity:
- Cosine similarity scores are computed between the user's query and each project's data.
- Higher scores indicate better matches.

### 4. Returning Results:
- Projects are ranked by similarity scores.
- The backend responds to the frontend with project IDs and similarity scores.

---

## Technology Stack

- **Programming Language**: Python
- **Framework**: Flask
- **NLP Library**: NLTK
- **Vectorization**: TF-IDF
- **API Integration**: External API for project data
- **Cross-Origin Requests**: Flask-CORS

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/procollab-backend.git
   cd procollab-backend
## Installation and Setup

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python app.py
![deployment image](https://drive.google.com/uc?export=view&id=1hHSPSxyzsbHvwPoo4l2sj2OBFUdsOcKk)