import numpy as np
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to convert DNA sequence into k-mers
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Load and preprocess the data
human_data = pd.read_table('human_data.txt')
chimp_data = pd.read_table('chimp_data.txt')
dog_data = pd.read_table('dog_data.txt')

# Apply k-mer transformation
human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
human_data = human_data.drop('sequence', axis=1)
chimp_data['words'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp_data = chimp_data.drop('sequence', axis=1)
dog_data['words'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)
dog_data = dog_data.drop('sequence', axis=1)

# Convert k-mers to string sentences
human_texts = [' '.join(kmers) for kmers in human_data['words']]
chimp_texts = [' '.join(kmers) for kmers in chimp_data['words']]
dog_texts = [' '.join(kmers) for kmers in dog_data['words']]

# Combine human, chimp, and dog data
X = human_texts + chimp_texts + dog_texts
y = np.concatenate([human_data['class'].values, chimp_data['class'].values, dog_data['class'].values])

# Vectorize the k-mers (Convert the DNA k-mers to a numerical format for ML)
cv = CountVectorizer(ngram_range=(4, 4))  # Adjust 'ngram_range' for different k-mer sizes
X = cv.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Naive Bayes Classifier)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Class label mapping
class_labels = {
    0: "G protein coupled receptors",
    1: "Tyrosine kinase",
    2: "Tyrosine phosphatase",
    3: "Synthetase",
    4: "Synthetase",
    5:"Ion channel",
    6:"Transcription factor"
    # Add more mappings based on your dataset classes
}


# Define a prediction function for Gradio interface
def predict_dna_function(sequence):
    kmers = getKmers(sequence)
    kmers = ' '.join(kmers)
    vectorized_sequence = cv.transform([kmers])
    prediction = model.predict(vectorized_sequence)
    class_description = class_labels.get(prediction[0], "Unknown class")
    return f"Predicted gene function class: {prediction[0]} - {class_description}"

# Set up Gradio interface
interface = gr.Interface(
    fn=predict_dna_function,
    inputs=gr.Textbox(lines=10, label="Input DNA Sequence"),
    outputs=gr.Textbox(label="Prediction"),
    title="DNA Sequence Classifier",
    description="Enter a DNA sequence to predict its gene function class based on k-mer analysis."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
