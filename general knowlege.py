import tkinter as tk
from tkinter import scrolledtext
import random
import nltk
import string 
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity 
from nltk.stem import WordNetLemmatizer

#download nltk data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

corpus = [
    "The study of universe is known as Cosmology."

    "The universe is commonly defined as the totality of everything that exists including all physical matter and energy, the planes, starts galaxies and the contents of intergalactic space."

    "Galaxy – A galaxy is a vast system of billions of starts, dust and light gases bound by their own gravity. " "There are 100 billion galaxies in the universe and each galaxy has, on average, 100 billion stars."

    "Our galaxy is Milky Way Galaxy (or the Akash Ganga) formed after the Bib Bang."

    "Andromeda is the nearest galaxy to the Milky Way."

    "The Big Bang Theory – Big Bang was an explosion of concentrated matter in the universe that occurred 15 billion years ago, leading to the formation of galaxies of stars and other heavenly bodies."

    "It is believed that universe should be filled with radiation called the “cosmic microwave background.” NASA has launched two missions these radiation, i.e. the Cosmic Background Explorer (COBA) and the Wilkinson Microwave Anisotropy Probe (WMAP)."

    "Stars are heavenly bodies made up of hot burning gases and they shine by emitting their own light."

    "Black Hole Stars having mass greater than three times that of the Sun, have very high gravitational power, so that even light cannot escape from its gravity and hence called black hole."

   " Comets Made up of frozen gases. They move around the Sun in elongated elliptical orbit with the tail always pointing away from the Sun."
   "There are a total of sixteen districts in Manipur. Until 2016, there were only nine districts in the state. However, the Manipur Cabinet announced that seven more districts would be created in the state by bifurcating some of the existing hilly districts of Manipur."
   "In terms of area, the largest district of Manipur is Churachandpur. On the other hand, the smallest district of Manipur is Bishnupur."
   "Biggest Planet- – Jupiter",
    "Biggest Satellite – Ganymede (Jupiter)",
    "Blue Planet – Earth",
"Green Planet – Uranus",
"Brightest Planet – Venus",
"Brightest Planet outside Solar System – Sirius (Dog Star)",
"Closet Star of Solar System – Proxima Centauri",
"Coldest Planer – Netune",
"Evening Star – Venus",
"Farthest Planet from Sun – Neptune",
"Planet with maximum in number of satellites – Jupiter",
"Fastest revolution in Solar System – Mercury",
"Hottest Planet – Venus",
"Densest Planet – Earth",
"Fastest rotation in Solar System – Jupiter",
"Morning Star – Venus",
"Nearest Planet to the Earth – Venus",
"Nearest Planet to Sun – Mercury",
"Red Planet – Mars",
"Slowest Revolution in Solar System – Neptune",
"Slowest Rotation in Solar System – Venus",
"Smallest Planet – Mercury",
"Smallest Satellite – Deimos (Mars)",
"Earth’s Twin – Venus",
"Only Satellite with an atmosphere like Earth – Titan"
]

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return " ".join(tokens)

corpus = [preprocess(sentence) for sentence in corpus]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    idx = similarity.argsort()[0][-1]
    if similarity[0][idx] > 0.2:
        return corpus[idx]
    else:
        return "Sorry, I didn't understand that. Can you rephrase?"

# Chat loop
print("Chatbot: Hello! Ask me anything. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(user_input))