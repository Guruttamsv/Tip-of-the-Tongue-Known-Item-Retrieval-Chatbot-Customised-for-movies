# Tip of the Tongue (TOT) Movie Retrieval Chatbot

This project is a Tip-of-the-Tongue (TOT) known-item retrieval chatbot, designed to assist users in identifying movies they cannot fully recall. The system uses a conversational interface powered by LLama2 and NLP techniques to iteratively narrow down movie options based on vague or partial descriptions provided by users. The system utilizes autoencoder anomaly detection to extract unique features from movie plots and generates targeted questions to help refine the search.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

The TOT chatbot assists users in retrieving movies by:

1. Accepting movie descriptions from users.
2. Extracting movie options using the LLama2 language model.
3. Fetching movie plots from Wikipedia.
4. Identifying unique plot details using an autoencoder-based anomaly detection system.
5. Generating questions to help the user confirm or reject movie candidates.
6. Concluding with the movie title once enough information has been collected.

## Features

* Conversational Interface: A user-friendly chatbot interface built with Tkinter to allow easy interactions.
* Large Language Model: Utilizes LLama2 for text generation, question generation, and brainstorming potential movie options.
* Autoencoder for Anomaly Detection: Differentiates movie plots and identifies unique plot points for generating questions.
* Wikipedia Plot Retrieval: Automatically searches for and extracts movie plot summaries from Wikipedia.
* Interactive Questioning: Dynamically generates yes/no questions to refine movie choices based on the user's responses.

## System Requirements

+ **Python**: 3.8+
+ **GPU**: High-performance GPU recommended for faster model loading and execution.
+ **Libraries**: LangChain, HuggingFace, Transformers, Wikipedia API, Keras, TensorFlow, NLTK, and others (see below for installation).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/tot-movie-chatbot.git
cd tot-movie-chatbot
```
2. Set up a virtual environment: Using conda or virtualenv is recommended:
```bash
conda create -n tot-chatbot python=3.8
conda activate tot-chatbot
```
3. Install required packages: Install the necessary dependencies:
```bash
pip install langchain transformers llama-index huggingface-cli accelerate bitsandbytes wikipedia-api googlesearch-python
```
4. Login to Hugging Face: Obtain a token from Hugging Face and authenticate:
```bash
huggingface-cli login
```
5. Download models: This project uses the Llama2 model from Hugging Face. You will need access to the meta-llama/Llama-2-70b-chat-hf model, which requires permissions:
```bash
huggingface-cli download meta-llama/Llama-2-70b-chat-hf
```
6. Run the chatbot: Once all packages are installed and models are downloaded, run the Python script:
```bash
python chatbot.py
```
## Usage

Upon launching the program, the chatbot will ask the user to describe the movie they are trying to recall.
The system will process the description, generate a list of suspected movies, and then prompt the user with questions based on the extracted plot anomalies.
The chatbot will continue to refine the movie options through user responses until it identifies the correct movie.
Final movie titles are presented with a fun response including emojis.

## Project Structure

├── chatbot.py            # Main program file containing the chatbot implementation

└──  README.md            # Project documentation

## Limitations and Future Work

### Limitations:
High GPU Requirement: The model is large and requires significant computing resources for smooth execution.
Autoencoder Accuracy: Performance is highly dependent on the quality of the anomaly detection in plot differences, which may struggle with highly similar or complex plots.
Limited Movie Dataset: The system relies on Wikipedia movie plots, which may not always be accurate or up-to-date.

### Future Work:
Extend the system to handle non-movie TOT retrievals, such as books or TV shows.
Implement a larger, more diverse dataset for improved movie suggestions.
Optimize the autoencoder architecture for better anomaly detection with complex movie plots.

## Acknowledgements

This project was developed as part of a BSc in Computer Science at Brunel University London by Guruttam Sivakumar Vijayalakshmi. Special thanks to Dr. Matloob Khushi, Dr. Mark Perry, and Dr. Fang Wang for their guidance and feedback.

Additionally, thanks to the developers and contributors at Hugging Face, LangChain, and Llama2 for providing the necessary tools and libraries to make this project a reality.

