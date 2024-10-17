
"""--------------------------------------All Imports--------------------------------------"""
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import locale
locale.getpreferredencoding = lambda *args, **kwargs: "UTF-8"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BertTokenizer, BertModel
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import re
import requests
import random
import wikipediaapi
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Masking
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# Download the 'punkt' resource
nltk.download('punkt')
nltk.download('stopwords')
from langchain.text_splitter import RecursiveCharacterTextSplitter
from googlesearch import search
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Dense  # If you need Dense again for some specific purpose
from urllib.parse import urlparse
import sys
import tkinter as tk

print("IMPORTS COMPLETED SUCCESSFULLY!!!")

"""--------------------------------------------------------------------------------------"""


def display(message, speaker):
    chat_text.config(state=tk.NORMAL)
    chat_text.insert(tk.END, f"{speaker}: {message}\n")
    chat_text.config(state=tk.DISABLED)
    chat_text.yview(tk.END)

def entry():
    input_frame.pack(side=tk.BOTTOM, pady=10)
    window.wait_variable(user_input_var)
    user_input = user_entry.get()
    display(user_input, "User")
    user_entry.delete(0, tk.END)  # Clear the text in the entry box
    input_frame.pack_forget()  # Hide the input frame after submitting
    return user_input

# Function to handle the Submit button click
def get_user_input():
    user_input_var.set(1)

# Create the main window
window = tk.Tk()
window.title("TOT Chatbot Window")

# Create and place widgets in the window
chat_text = tk.Text(window, height=10, width=40, state=tk.DISABLED)
chat_text.pack(pady=10, expand=True, fill=tk.BOTH)  # Make the text widget flexible

# Frame for user input section
input_frame = tk.Frame(window)
user_entry = tk.Entry(input_frame, width=30)
user_entry.pack(side=tk.LEFT)
submit_button = tk.Button(input_frame, text="Submit", command=get_user_input)
submit_button.pack(side=tk.LEFT)

# Variable to control the user input
user_input_var = tk.IntVar()

# Make the window flexible and maximize it
window.geometry("800x600")  # Set an initial size
window.attributes('-zoomed', True)  # Maximize the window


"""--------------------------------------------------------------------------------------"""

"""----------------------------------------Set Up-----------------------------------------"""
# Import necessary libraries and modules
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

# Load the pre-trained model for causal language modeling
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             load_in_4bit=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.float16)

# Create a text generation pipeline using the loaded model and tokenizer
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

# Create a Hugging Face Pipeline for text generation with specific parameters
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.5, 'max_length': 150, 'top_k': 2})

# Load tools and set up memory for the conversational agent
tools = load_tools([], llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history")
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=False,
    max_iterations=5,
    memory=memory,
    handle_parsing_errors=True
)


print("SET-UP COMPLETED SUCCESSFULLY!!!")

"""--------------------------------------------------------------------------------------"""
"""-----------------------------------Global variables-----------------------------------"""

prompt = ""

ToT_suspected_movies = []

suspected_movies_plot = {}

ToT_question_bank = []

ToT_answer = ""

plot_vect = None

"""--------------------------------------------------------------------------------------"""

"""----------------------------------------Phase 1----------------------------------------"""

"""Helper Functions----------------------------------------"""

# Define a function to extract a substring between "$" and "@"
def slice(sentence):
    a1 = 0
    b1 = 0
    for i in range(0, len(sentence) - 1):
        if sentence[i] == "$":
            a1 = i
        elif sentence[i] == "@":
            b1 = i
            break
    return sentence[a1 + 1: b1 - 1]

# Define a function to extract movie names and years from a description
def info_from_corpus(description):
    names = re.findall(r'"([^"]*)"', description)
    years = re.findall(r'\((\d{4})\)', description)
    combined = list(set([f'{name} ({year})' for name, year in zip(names, years)]))
    return combined

"""Main Functions----------------------------------------"""

# Define a function for generating the opening question prompt
def Generated_text_1():
    a = "you are a movie finder bot (item retrieval), u need to ask the user to give all the details they know of the movie they are looking for. present your opening question with a greeting."
    first_prompt = PromptTemplate(
        input_variables=["Question"],
        template="""form one question based on the use case :
        Make sure you alter the question such that it is more fun for the user to read,
        no specifics are asked, the user should feel comfortable giving their own answer,
        use emoji's if needed,
        begin with $$$ and end with @@@
        should not include asking the name of the movie
        question should be short:
        {Question} """
    )

    first_extraction_chain = LLMChain(llm=llm, prompt=first_prompt)

    pre_reply = first_extraction_chain.run(a)
    reply = slice(pre_reply)
    return reply

# Define a function for generating the follow-up question prompt
def Generated_text_2():
    a = "you are a movie finder bot (item retrieval), u asked the user to give all the details they know of the movie they are looking for. They have answered. now you need to ask if they can remember anything else"
    first_prompt = PromptTemplate(
        input_variables=["Question"],
        template="""form one question based on the use case :
        You do not have to mention you are a movie finder bot
        Make sure you alter the question such that it is more fun for the user to read,
        no specifics are asked, the user should feel comfortable giving their own answer,
        use emoji's if needed,
        begin with $$$ and end with @@@
        should not include asking the name of the movie
        question should be short:
        {Question}"""
    )

    first_extraction_chain = LLMChain(llm=llm, prompt=first_prompt)

    pre_reply = first_extraction_chain.run(a)
    reply = slice(pre_reply)
    return reply

# Define a function to extract movie information from user's input
def extract_movies_from_llm(prompt):
    suggestions = ""

    # Get movie suggestions from the conversational agent
    response = conversational_agent("""Find me a movie that best matches the description.
    In your response, give me the movie in this format {"movie title" (year of release)}: example - 'Bad Boys' (1990).
    Movie Description : """ + prompt)
    suggestions = suggestions + response['output']

    for i in range(4):
        # Ask for additional movie suggestions
        response = conversational_agent("brainstorm other movies that even slightly match the provided Movie Description, apart from above suggestions. give me the movie in the same format.")
        suggestions = suggestions + response['output']
    return info_from_corpus(suggestions)


print("PHASE 1 COMPLETED SUCCESSFULLY!!!")

"""--------------------------------------------------------------------------------------"""

"""----------------------------------------Phase 2----------------------------------------"""

"""Helper Functions----------------------------------------"""

# Define a function to perform a Google search and return the Wikipedia link for a given query
def search_google(query):
    try:
        search_results = list(search(query + " wikipedia", num_results=1))
        return search_results[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Define a function to get the plot summary from a Wikipedia URL
def get_wikipedia_plot_from_url(url):
    parsed_url = urlparse(url)
    page_title = parsed_url.path.split("/")[-1]

    wiki_wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'my_movie_app/1.0'})

    page_py = wiki_wiki.page(page_title)
    if page_py.exists():
        plot_section = None
        for section in page_py.sections:
            if "plot" in section.title.lower(): 
                plot_section = section
                break
        return plot_section.text

# Define a function to generate explanations for items using the Language Model

"""Vectorizing data"""

# Define a function to vectorize movie plots using BERT embeddings
def vectorize_plots(suspected_movies_plot):
    vect_movie_plots = []
    for value in suspected_movies_plot.values():
        tokens = tokenizer.encode(value, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            vect_movie_plots.append(model(tokens)[0])
    return vect_movie_plots

# Define a function to vectorize a given prompt using BERT embeddings
def vectorize_prompt(prompt):
    tokens = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        vect_prompt = (model(tokens)[0])
    return vect_prompt

"""Get Outliers"""

# Define a function to clean and preprocess text data
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Define a function to chunk train data for autoencoder
def chunk_train_data(text_data, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0
    )
    docs = text_splitter.create_documents(text_data)
    chunks = []
    for i in docs:
        chunks.append(i.page_content)
    return chunks

# Define a function to chunk test data for autoencoder
def chunk_test_data(text_data, chunk_size):
    chunks = [text_data[i:i + chunk_size] for i in range(0, len(text_data), chunk_size)]
    return chunks

# Define a function to preprocess data for autoencoder training
def preproccess_data(key_at_index, movies):
    test = movies[key_at_index]
    train = [value for key, value in movies.items() if key != key_at_index]

    cleaned_texts = [clean_text(text) for text in train]

    chunk_size = 100
    text_chunks = chunk_train_data(cleaned_texts, chunk_size)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_chunks)

    train_sequences = tokenizer.texts_to_sequences(text_chunks)
    train_data = pad_sequences(train_sequences)

    test_text_cleaned = clean_text(test)
    test_data = chunk_test_data(test_text_cleaned, 100)

    return (test_data, train_data, train_sequences)

# Define a function to find sentences containing anomalies
def find_sentences(test, anomaly):
    sent = []

    result_string = []
    corpwords = test.lower().split()
    for phrase in anomaly:
        target_words = phrase.split()
        index = []

        for target_word in target_words:
            indices = [index for index, c_word in enumerate(corpwords) if target_word in c_word]
            index.append(indices)

        while index and not index[0]:
            del index[0]

        while index and not index[-1]:
            del index[-1]

        if len(index[0]) == 1 and len(index[-1]) == 1:
            a = index[0][0]
            b = index[-1][0]
        else:
            ListS = [(i, j) for i in index[0] for j in index[-1] if
                     all(any(i <= num <= j for num in sublist) if sublist else True for sublist in index)]
            a = max(sublist[0] for sublist in ListS) if ListS else None
            b = min(sublist[1] for sublist in ListS) if ListS else None
        result_string = result_string + ' '.join(corpwords[a:b]).split('.')
    filtered_result = [value for value in result_string if len(value.split()) > 1]

    def paragraph_to_sentences(paragraph):
        sentences = nltk.sent_tokenize(paragraph)
        return sentences

    sentences = paragraph_to_sentences(test)

    for result in filtered_result:
        for i, sentence in enumerate(sentences, 1):
            if result.strip() in sentence.lower():
                if sentence not in sent:
                    sent.append(sentence)
                    break
    return sent

# Define a function to run the autoencoder and detect anomalies
def run_autonecoder(movies):
    result = []
    tokenizer = Tokenizer()
    for i in range(0, len(movies)):
        keys_list = list(movies.keys())
        key_at_index = keys_list[i]

        test_raw, train_data, train_sequences = preproccess_data(key_at_index, movies)
        max_len = max(len(seq) for seq in train_sequences)
        input_size = train_data.shape[1]

        model = Sequential([
            Masking(mask_value=0.0, input_shape=(input_size,)),
            Dense(64, activation='relu'),
            Dense(input_size, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy')
        # Train the autoencoder
        model.fit(train_data, train_data, epochs=10, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)
        anomaly = []
        #
        for j in range(0, len(test_raw)):
            test_string = test_raw[j]
            test_sequence = tokenizer.texts_to_sequences([test_string])
            test_data = pad_sequences(test_sequence, padding='post', maxlen=max_len)

            # Detect Anomalies
            encoded_test_data = model.predict(test_data, verbose=0)

            # Calculate reconstruction error (binary crossentropy)
            bce = np.mean(tf.keras.losses.binary_crossentropy(test_data, encoded_test_data))

            # Set a threshold for anomaly detection
            threshold = 0.1  # can adjust this to fix anomaly

            # Check if the reconstruction error is above the threshold
            is_anomaly = bce > threshold

            if (is_anomaly):
                anomaly.append(test_raw[j])
        result = result + find_sentences(movies[key_at_index], anomaly)
    return result

"""Main Functions----------------------------------------"""

# Define a function to get plots for movies by performing Google searches and fetching Wikipedia plot summaries
def get_plots_for_movies(movies):
    plot = {item: None for item in movies}

    for key in plot:
        url = search_google(key)
        print(url)
        plot[key] = get_wikipedia_plot_from_url(url)

    return plot

# Define a function to process data using the autoencoder
def process_data(plot):
    questions = run_autonecoder(plot)
    random.shuffle(questions)
    return questions



print("PHASE 2 COMPLETED SUCCESSFULLY!!!")

"""--------------------------------------------------------------------------------------"""
"""----------------------------------------Phase 3----------------------------------------"""

"""Helper Functions----------------------------------------"""

# Define a function to compare the vectorized prompt with the vectorized movie plots
def compare(prompt, vect_movie_plots):
    # Vectorize the input prompt
    vect_prompt = vectorize_prompt(prompt)
    closeness = []
    
    # Iterate over each movie plot
    for movie in vect_movie_plots:
        # Extract the BERT embeddings from the movie plot tensor
        X = movie[0].numpy()
        # Extract the BERT embeddings from the vectorized prompt tensor
        Y = vect_prompt[0].numpy()

        # Flatten the tensors
        X_flat = X.reshape((X.shape[0], -1))
        Y_flat = Y.reshape((Y.shape[0], -1))

        # Normalize the flattened tensors
        X_normalized = X_flat / np.linalg.norm(X_flat, axis=1, keepdims=True)
        Y_normalized = Y_flat / np.linalg.norm(Y_flat, axis=1, keepdims=True)

        # Calculate cosine similarity between the normalized tensors
        similarity_percentage = cosine_similarity(X_normalized, Y_normalized) * 100

        # Append the similarity percentage to the closeness list
        closeness.append(similarity_percentage[0][0])
    
    # Return the list of similarity percentages for each movie plot
    return closeness


"""Main Functions----------------------------------------"""

# Function to generate a question based on a given sentence
def Generated_question(question):
    first_prompt = PromptTemplate(
        input_variables=["Question"],
        template="""form one question based on the use case:
        use case: you are going to ask a yes or no question to the user based on the SENTENCE;
        below is a line from the plot of a movie;
        your job is to form a question asking if that scene is in the movie the user is looking for, such a way the answer can be either yes or no.
        for example: sentence - "They teamed up to kill the enemies."
        question should be like - "Was there a scene where the characters teamed up to defeat the enemy?" so the answer can either be yes or no
        Make sure you alter the question such that it is more fun for the user to read,
        if there is a name in the sentence, don't use the name just replace it with 'character',
        begin the question with $$$ and end with @@@
        should not include asking the name of the movie
        question should be short.
        SENTENCE : {Question} """
    )
    first_extraction_chain = LLMChain(llm=llm, prompt=first_prompt)
    pre_reply = first_extraction_chain.run(question)

    # Slice the generated reply to extract the final result
    reply = slice(pre_reply)
    return reply

# Function to filter options based on user answer and update the prompt
def filter_options(prompt, answer, question, v_plot):
    if answer.lower() == "yes":
        prompt += question
    percentage = compare(prompt, v_plot)
    print(percentage)

    # Find the maximum percentage and its index
    max_per = max(percentage)
    max_index = percentage.index(max_per)

    return [max_per >= 80, max_index], prompt

# Function to generate an answer sentence based on the user's answer
def Generated_answer(answer):
    first_prompt = PromptTemplate(
        input_variables=["Answer"],
        template="""you are a sentence generator
        Answer is {Answer}
        This is the answer(movie name) that the user has been looking for.
        generate a single fun sentence with emoji's saying 'the answer has been found' and reveal the answer.
        begin the sentence with $$$ and end with @@@
        """
    )

    first_extraction_chain = LLMChain(llm=llm, prompt=first_prompt)
    pre_reply = first_extraction_chain.run(answer)

    # Slice the generated reply to extract the final result
    reply = slice(pre_reply)
    return reply



print("PHASE 3 COMPLETED SUCCESSFULLY!!!")


"""--------------------------------------------------------------------------------------"""

"""-----------------------------------Actual Implementation-----------------------------------"""
#Phase 1
print("PHASE 1 IMPLEMENTATION IN PROCCESS")
display(Generated_text_1())
user_input = entry()
display(Generated_text_2())
follow_up = entry()
prompt = user_input+follow_up
ToT_suspected_movies = extract_movies_from_llm(prompt)

#Phase 2
print("PHASE 2 IMPLEMENTATION IN PROCCESS")
suspected_movies_plot = get_plots_for_movies(ToT_suspected_movies)
ToT_question_bank = process_data(suspected_movies_plot)

#Phase 3
print("PHASE 3 IMPLEMENTATION IN PROCCESS")
concluded = [False, None]
display("Answer yes or no:")
v_plot = vectorize_plots(suspected_movies_plot)
while not concluded[0]:
  question = ToT_question_bank.pop()
  print(Generated_question(question))
  response = entry()
  concluded, prompt = filter_options(prompt, response, question, v_plot)
ToT_answer = ToT_suspected_movies[concluded[1]]
display(Generated_answer(ToT_answer))