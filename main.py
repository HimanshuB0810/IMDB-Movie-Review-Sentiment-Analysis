# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# Load the pre-trained model 
model=load_model("simple_RNN_imdb.h5")

# Step 2: Helper Functions
# Function to decode the review
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## Streamlit App

import streamlit as st
import time

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("About")
    st.info(
        "This app uses a machine learning model to classify a movie "
        "review as either **Positive** or **Negative**."
    )
    st.header("How It Works")
    st.write(
        "1. Enter a movie review in the text box.\n"
        "2. Click the 'Analyze Sentiment' button.\n"
        "3. The app will display the sentiment and a confidence score."
    )


# --- MAIN APP ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below to classify its sentiment as either **Positive** or **Negative**.")

# Using a form prevents the app from rerunning on every interaction
with st.form(key='review_form'):
    # User Input
    user_input = st.text_area(
        "Enter your movie review here:",
        height=150,
        placeholder="e.g., 'This movie was absolutely fantastic! The acting was superb and the plot was gripping.'",
        key='user_input' # Use a key to access the value outside the form
    )

    # Submit button
    submit_button = st.form_submit_button(label='Analyze Sentiment', use_container_width=True)


# --- ANALYSIS AND DISPLAY LOGIC ---
if submit_button:
    if user_input:
        # Simulate processing time and show a spinner
        with st.spinner('Analyzing your review...'):
            time.sleep(1) # Simulate model latency
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            prediction_score = prediction[0][0]

        st.subheader("Analysis Result")
        
        # Display sentiment with custom styling
        if prediction_score > 0.5:
            sentiment = "Positive"
            st.success(f"**Sentiment: {sentiment}** 👍")
        else:
            sentiment = "Negative"
            st.error(f"**Sentiment: {sentiment}** 👎")

        # Display results in columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            # Use st.metric for a nice visual representation of the score
            st.metric(
                label="Confidence Score",
                value=f"{prediction_score:.2%}",
                help="This is the model's confidence in its prediction."
            )
        
        with col2:
            # Add a progress bar for another visual cue
            st.write("Sentiment Score Breakdown:")
            st.progress(float(prediction_score))

        # Show the original review in an expander
        with st.expander("Show Original Review"):
            st.write(user_input)

    else:
        # Show a warning if the user clicks the button with no input
        st.warning("Please enter a movie review to analyze.")

else:
    st.info("Enter a review and click 'Analyze Sentiment' to see the magic happen!")
