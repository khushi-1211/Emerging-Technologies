import os
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
import re
import pickle
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download required NLTK data
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

class EnhancedNLPChatBot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.knowledge_base = {}
        self.responses = []
        self.response_vectors = None
        self.svd = None
        self.similarity_threshold = 0.15
        self.context_window = 3
        self.book_summaries = {}

    def train(self, training_files):
        """Enhanced training method with NLTK-based processing."""
        all_text = []
        book_contents = defaultdict(list)

        for file_path in training_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    book_title = os.path.basename(file_path).replace('.txt', '').replace('_', ' ').title()
                    
                    self.knowledge_base[book_title] = content
                    self.book_summaries[book_title] = self.generate_book_summary(content)
                    sentences = sent_tokenize(content)
                    
                    for i in range(len(sentences)):
                        context_window = sentences[max(0, i - self.context_window):i + self.context_window + 1]
                        processed_text = self.preprocess_text(' '.join(context_window))
                        if len(processed_text.split()) > 3:
                            all_text.append(processed_text)
                            book_contents[book_title].append(sentences[i])
                            self.responses.append(sentences[i])
                    
                logging.info(f"Processed {book_title}: {len(book_contents[book_title])} sentences extracted")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
        
        if all_text:
            self.response_vectors = self.vectorizer.fit_transform(all_text)
            self.svd = TruncatedSVD(n_components=100)
            self.response_vectors = self.svd.fit_transform(self.response_vectors)
            logging.info(f"Training completed. Processed {len(all_text)} text segments")
        else:
            logging.warning("No text was successfully processed for training!")

    def preprocess_text(self, text):
        """Enhanced text preprocessing using NLTK."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        words = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words and word.isalnum()
        ]
        return ' '.join(tokens)

    def extract_key_phrases(self, text):
        """Extract important phrases using NLTK's POS tagging."""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        key_phrases = []
        current_phrase = []
        
        for word, tag in pos_tags:
            if tag.startswith(('NN', 'NNP', 'JJ')):
                current_phrase.append(word)
            else:
                if current_phrase:
                    key_phrases.append(' '.join(current_phrase))
                    current_phrase = []
        
        if current_phrase:
            key_phrases.append(' '.join(current_phrase))
        
        return list(set(key_phrases))

    def generate_book_summary(self, content):
        """Generate a summary using key sentence extraction."""
        sentences = sent_tokenize(content[:5000])
        sentence_scores = {}
        
        for sentence in sentences:
            key_phrases = self.extract_key_phrases(sentence)
            sentence_scores[sentence] = len(key_phrases)
        
        important_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return ' '.join(sent for sent, score in important_sentences)

    def get_response(self, user_input):
        """Enhanced response generation with context awareness."""
        if not self.responses or self.response_vectors is None:
            return "I haven't been trained yet!"

        cleaned_input = self.preprocess_text(user_input)
        book_query_patterns = [
            r"what happens in (.+?)\?",
            r"tell me about (.+?)",
            r"summarize (.+?)",
            r"what is (.+?) about\?"
        ]

        for pattern in book_query_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                query_subject = match.group(1)
                for book_title in self.knowledge_base.keys():
                    if query_subject in book_title.lower():
                        return self.book_summaries[book_title]

        user_vector = self.vectorizer.transform([cleaned_input])
        user_vector = self.svd.transform(user_vector)
        similarities = cosine_similarity(user_vector, self.response_vectors)

        top_indices = np.argsort(similarities[0])[-5:][::-1]

        for idx in top_indices:
            if similarities[0][idx] > self.similarity_threshold:
                response = self.responses[idx]
                context_start = max(0, idx - 1)
                context_end = min(len(self.responses), idx + 2)
                if context_end - context_start > 1:
                    response = ' '.join(self.responses[context_start:context_end])
                return response

        default_responses = [
            f"I understand you're asking about {cleaned_input}, but I need more context to provide a relevant answer.",
            "Could you please rephrase your question? I want to make sure I understand correctly.",
            "I'm not confident about answering that specific question. Could you ask it differently?",
            "That's an interesting question. Could you provide more details about what you'd like to know?",
        ]
        return np.random.choice(default_responses)

    def save_model(self, filepath='enhanced_chatbot_model.pkl'):
        """Save the trained model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Model saved as {filepath}")

    @classmethod
    def load_model(cls, filepath='enhanced_chatbot_model.pkl'):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    chatbot = EnhancedNLPChatBot()
    
    training_files = [
        "static/books/Alice_In_Wonderland.txt",
        "static/books/Frankenstein.txt",
        "static/books/Moby_Dick.txt",
        "static/books/Pride_And_Prejudice.txt",
        "static/books/Great_Gatsby.txt"
    ]

    logging.info("Training the enhanced chatbot...")
    chatbot.train(training_files)
    
    chatbot.save_model()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = chatbot.get_response(user_input)
        print("Bot:", response)
