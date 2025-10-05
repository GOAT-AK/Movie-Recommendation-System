# Movie-Recommendation-System
This project focuses on recommending movies to users based on their preferences using Collaborative Filtering (User-based &amp; Item-based). The system learns from user rating patterns to suggest movies that similar users or similar items highly rated.

# Table of Content

* [Brief](#Brief)  
* [DataSet](#DataSet)  
* [How_It_Works](#How_It_Works)  
* [Tools](#Tools)  
* [Model_Performance](#Model_Performance)  
* [Remarks](#Remarks)  
* [Usage](#Usage)  
* [Sample_Run](#Sample_Run)



# Brief

With the rapid expansion of streaming platforms and movie databases, **recommender systems** have become essential in helping users discover content that aligns with their preferences.  
This project implements a **Collaborative Filtering Recommendation System** using both **User-based** and **Item-based similarity models**.  

The system suggests top-rated movies that a user has not yet seen, based on the ratings and similarities among users and movies.



# DataSet

The dataset used in this project is the **[MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)**, which includes 100,000 ratings from 943 users on 1,682 movies.  

Each user has rated at least 20 movies, and demographic information (age, gender, occupation, zip) is also available.  
However, this project focuses primarily on the **ratings** and **movie metadata** for collaborative filtering.



### Files Used

| File Name | Description |
|------------|-------------|
| `u1.base` | Training data (80% of full dataset) containing user–movie ratings. |
| `u1.test` | Test data (20% of full dataset) used for evaluation. |
| `u.item` | Movie metadata including title, release date, genres, and IMDb links. |
| `user_similarity.pkl` | Precomputed user-user similarity matrix. |
| `item_similarity.pkl` | Precomputed item-item similarity matrix. |



### Columns in `u1.base`

| Attribute | Description |
|------------|-------------|
| user_id | Unique ID for each user (1–943). |
| item_id | Unique ID for each movie (1–1682). |
| rating | Rating score (1–5). |
| timestamp | Unix timestamp of when the rating was given. |




# How_It_Works

- Load and preprocess the **MovieLens 100K** dataset.  
- Construct a **User-Item Rating Matrix** where rows = users and columns = movies.  
- Compute **User Similarity** (User-based CF) and **Item Similarity** (Item-based CF) using **Cosine Similarity**.  
- Predict ratings for unseen movies using weighted averages from similar users/items.  
- Recommend the **Top-N** movies with the highest predicted ratings.  
- Evaluate model performance using **Precision@K** on the test set.  
- Save the trained similarity matrices (`user_similarity.pkl`, `item_similarity.pkl`) for deployment.  
- Deploy a **Streamlit web app** where users can select a user ID, choose between user/item-based recommendation, and view predicted movies in real-time.




# Tools & Libraries

I. Jupyter Notebook & VS Code  
II. Python 3.x  
III. pandas, numpy  
IV. matplotlib, seaborn  
V. scikit-learn  
VI. pickle  
VII. Streamlit



# Model_Performance

Both collaborative filtering models were evaluated using **Precision@5** on the test data.

| Model Type | Precision@5 |
|-------------|--------------|
| User-based Collaborative Filtering | 0.1011 |
| Item-based Collaborative Filtering | 0.5102 |

The results show that the **Item-based model** provides better recommendations in this dataset, as it effectively captures relationships between movies based on shared user ratings.



# Remarks

* This Python project was developed and tested in **Jupyter Notebook**.  
* Ensure the required dependencies are installed by running:

  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn streamlit


# Usage

To begin utilizing this application, follow these steps:

1. Clone this repository:
   
   ```bash
   git clone https://github.com/GOAT-AK/Movie-Recommendation-System

2. Navigate to the cloned repository:

   ```bash
   cd Movie-Recommendation-System

3. Run the Jupyter Notebook:

   ```bash
   Movie_R.ipynb

4. Launch the Streamlit app:
   
   ```bash
   streamlit run Script_R.py



# Sample_Run


* Pic 1

<img width="1042" height="654" alt="Image" src="https://github.com/user-attachments/assets/ded84ee9-f1cc-47b4-be25-6c1cb24e4e0a" />


