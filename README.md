# IMDB Movie Review Sentiment Analysis 🎬

This project is a web application that classifies movie reviews as either **positive** or **negative** using a Simple Recurrent Neural Network (RNN) model built with TensorFlow and Keras. The application is built with Streamlit, providing an easy-to-use interface for users to enter a review and see the model's prediction.

## Features ✨

  * **Sentiment Analysis:** Classifies movie reviews into "Positive" or "Negative" categories.
  * **Confidence Score:** Displays the model's confidence in its prediction.
  * **Interactive Web App:** A user-friendly interface built with Streamlit.
  * **Simple RNN Model:** Utilizes a straightforward Recurrent Neural Network for the classification task.

## How It Works 🤔

The application uses a trained Simple RNN model to predict the sentiment of a movie review. Here is a brief overview of the process:

1.  **Input:** The user enters a movie review into a text area.
2.  **Preprocessing:** The text is cleaned and converted into a numerical format that the model can understand. This involves tokenizing the text, converting words to integers based on a pre-existing vocabulary, and padding the sequence to a fixed length.
3.  **Prediction:** The preprocessed input is fed into the trained Simple RNN model, which outputs a prediction score between 0 and 1.
4.  **Output:**
      * If the score is greater than 0.5, the sentiment is classified as **Positive**.
      * Otherwise, the sentiment is classified as **Negative**.
      * The application also displays the confidence score of the prediction.

## Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

1.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  **Enter a movie review** in the text box and click the "Analyze Sentiment" button to see the result.

## Technologies Used 💻

  * **TensorFlow:** For building and training the Simple RNN model.
  * **Streamlit:** For creating the interactive web application.
  * **NumPy:** For numerical operations.
  * **Scikit-learn:** For machine learning utilities.
  * **Pandas:** For data manipulation.