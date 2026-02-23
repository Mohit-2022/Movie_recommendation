                                    **Movie Recommendation System (End-to-End ML Project)**

  This project is an end-to-end Content-Based Movie Recommendation System that suggests similar movies to users based 
  on movie metadata such as genre, keywords, cast, crew, and overview.
  
  The project covers the complete Machine Learning lifecycle from:
  
  ✔ Data preprocessing
  ✔ Feature engineering
  ✔ Model building
  ✔ Similarity computation
  ✔ Model Seriallization
  ✔ Web app deployment using Streamlit

**2. Problem Statement**

  With the exponential growth of content on OTT platforms, users often face difficulty in selecting relevant movies based 
  on their preferences. This system helps users discover movies similar to the one they like by analyzing movie metadata 
  and computing similarity between movies.

**3. Project Workflow**

  a) Data Collection
  
        TMDB 5000 Movies Dataset
  
  b) Data Preprocessing
  
  c) Feature Engineering
  
        Text data merged into a single “tags” column
        
        Tokenization
        
        Stopword removal
        
        Stemming using NLTK
  
  d) Vectorization
  
        Applied Bag of Words using CountVectorizer
  
  e) Similarity Computation
  
        Cosine Similarity used to compute similarity between movies
  
  f) Model Serialization
  
        Saved using Pickle:
        
            movies_dict.pkl
            
            similarity.pkl
            
  g) Web Application
  
        Built interactive UI using Streamlit
        
        Users select a movie and receive top 5 similar movie recommendations

**4. Machine Learning Concepts Used**
    
  a) NLP based Feature Engineering
    
  b) Bag of Words Model
    
  c) Cosine Similarity
    
  d) Model Serialization (Pickle)
    
  e) Recommendation Systems

**5. Deployment**

The model has been deployed as a web application using Streamlit Cloud.

**6. Project Structure**

    movies_dict.pkl         → Model
    similarity.pkl          → Similarity model
    app.py                  → Streamlit application  
    requirements.txt        → Dependencies  
    README.md               → Project documentation
 
**7. Tech Stack**

    Python
    Pandas
    NLTK
    Similarity Cosine
    Streamlit
    Pickle

**8. Business Use Case**

  This type of recommendation engine can be used by:

  a) OTT Platforms (Netflix, Amazon Prime)
    
  b) Similar product Recommendations
    
  c) Music streaming apps
    
  d) News aggregators
    
    to improve user engagement and retention


[Click Here to Use Live App]
https://movierecommendation-f6iusaguhkv6riu34zsycs.streamlit.app/


Author
Mohit Kushwaha
LinkedIn: www.linkedin.com/in/mohit-kushwaha-024401112


