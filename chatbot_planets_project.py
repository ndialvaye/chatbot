import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Télécharger les ressources nécessaires de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# -------------------------
# Chargement et prétraitement des données
# -------------------------
def load_text():
    path = "planets.txt"
    if not os.path.exists(path):
        st.error("Fichier 'planets.txt' introuvable. Merci de le placer dans le même dossier que ce script.")
        st.stop()
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

# -------------------------
# Fonction de similarité
# -------------------------
def get_most_relevant_sentence(user_input, sentences, processed_sentences):
    user_input_clean = preprocess(user_input)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_sentences + [user_input_clean])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    index = np.argmax(cosine_sim)
    return sentences[index], cosine_sim[0][index]

# -------------------------
# Fonction du chatbot
# -------------------------
def chatbot(query, sentences, processed_sentences):
    if len(query.strip()) == 0:
        return "Veuillez entrer une question."
    response, score = get_most_relevant_sentence(query, sentences, processed_sentences)
    if score < 0.1:
        return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?"
    return response

# -------------------------
# Interface utilisateur Streamlit
# -------------------------
def main():
    st.set_page_config(page_title="Chatbot - Système Solaire", page_icon="🌍")
    st.title("🌌 Chatbot sur le Système Solaire")
    st.write("Pose-moi une question sur les planètes du système solaire ! Par exemple : *Quelle est la planète la plus chaude ?*")

    raw_text = load_text()
    sentences = [s.strip() + '.' for s in raw_text.split('.') if s.strip()]
    processed_sentences = [preprocess(s) for s in sentences]

    with st.form(key='chat_form'):
        question = st.text_input("Votre question :")
        submit_button = st.form_submit_button(label='Envoyer')

    if submit_button and question:
        answer = chatbot(question, sentences, processed_sentences)
        st.success("Réponse :")
        st.write(answer)

if __name__ == "__main__":
    main()
