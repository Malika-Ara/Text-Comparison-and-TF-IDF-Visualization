import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def get_user_input():
    paragraph1 = input("Enter Paragraph 1: ")
    paragraph2 = input("Enter Paragraph 2: ")
    return paragraph1, paragraph2

def summarize_paragraph(paragraph):
    sentences = sent_tokenize(paragraph)
    return ' '.join(sentences[:2])

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    words = nltk.word_tokenize(text)
    words = [ps.stem(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]

    return ' '.join(words)

def visualize_tfidf(paragraph, title):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([paragraph])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()[0]

    # Select top 10 terms with highest TF-IDF scores
    top_indices = tfidf_values.argsort()[-10:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    top_scores = [tfidf_values[i] for i in top_indices]

    # Bar graph for visualization
    plt.figure(figsize=(10, 5))
    plt.bar(top_terms, top_scores, color='skyblue')
    plt.title(title)
    plt.xlabel('Terms')
    plt.ylabel('TF-IDF Score')
    plt.show()

def compare_paragraphs(paragraph1, paragraph2):
    processed_paragraph1 = preprocess_text(paragraph1)
    processed_paragraph2 = preprocess_text(paragraph2)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_paragraph1, processed_paragraph2])

    # Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    cosine_similarity_value = similarity_matrix[0, 1]

    # Visualize TF-IDF for both paragraphs
    visualize_tfidf(processed_paragraph1, 'TF-IDF Visualization - Paragraph 1')
    visualize_tfidf(processed_paragraph2, 'TF-IDF Visualization - Paragraph 2')

    print("\nCosine Similarity:", cosine_similarity_value)

    if cosine_similarity_value > 0.6:
        print("\nParagraph 1 has similar objectives to Paragraph 2.")
    else:
        print("\nParagraph 1 does not have similar objectives to Paragraph 2.")

if __name__ == "__main__":
    paragraph1, paragraph2 = get_user_input()
    compare_paragraphs(paragraph1, paragraph2)
