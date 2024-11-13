import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd

# Loading the model and tokenizer
model = BertForSequenceClassification.from_pretrained('InnaK342/bert-toxic-comment')
tokenizer = BertTokenizer.from_pretrained('InnaK342/bert-toxic-comment')


def predict_toxicity(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).detach().tolist()
    return probabilities


# Streamlit web interface
st.title("Toxicity of comments")
user_input = st.text_area("Enter text:")

if st.button("Predict"):
    if user_input:
        results = predict_toxicity([user_input])
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        # Create a DataFrame for better visualization
        result_df = pd.DataFrame(results, columns=labels)

        # Display results as a table
        st.write("### Predicted Probabilities")
        st.write(result_df)
