import nltk
import spacy
import ssl
from textatistic import Textatistic
from textblob import TextBlob


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')  # Download stopwords if using NLTK
nltk.download('punkt')  # Download sentence tokenizer (required for readability)


def clean_text(text):
    text = text.lower()  # Lowercase
    stop_words = nltk.corpus.stopwords.words('english')  # Get stopwords (NLTK)
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    # Additional cleaning steps like stemming/lemmatization can be applied here
    return text


def calculate_readability(text):
    # Create a Textatistic object
    stats = Textatistic(text)

    # Get the Flesch Reading Ease score
    return stats.flesch_score


def count_keywords(text, keywords):
    count = 0
    for word in keywords:
        count += text.count(word)
    return count


def count_named_entities(text):
    doc = nlp(text)
    return len(list(doc.ents))  # Count named entities


def calculate_content_score(readability_score, word_count, keyword_count, named_entity_count):
    score = 0
    score += readability_score * weights['readability_score']
    score += word_count * weights['word_count']
    score += keyword_count * weights['keyword_count']
    score += named_entity_count * weights['named_entity_count']
    return score


# Read text from a file
with open("articles.csv", "r", encoding="utf-8") as file:
    text = file.read()

cleaned_text = clean_text(text)

readability_score = calculate_readability(cleaned_text)

word_count = len(cleaned_text.split())

# Define your relevant keywords here (replace with your specific terms)
relevant_keywords = ["data science", "machine learning", "artificial intelligence"]

keyword_count = count_keywords(cleaned_text, relevant_keywords)

nlp = spacy.load('en_core_web_sm')  # Load spaCy model

named_entity_count = count_named_entities(cleaned_text)

# Define weights for each feature based on your priorities (e.g., readability more important than length)
weights = {'readability_score': 0.3, 'word_count': 0.2, 'keyword_count': 0.4, 'named_entity_count': 0.1}


# Now you have the relevant scores and counts calculated based on the text read from the file.
content_score = calculate_content_score(readability_score, word_count, keyword_count, named_entity_count)
print(content_score)


# Perform sentiment analysis
blob = TextBlob(text)
sentiment = blob.sentiment

# Interpret the results
polarity = sentiment.polarity
if polarity > 0:
    print("Positive sentiment")
elif polarity < 0:
    print("Negative sentiment")
else:
    print("Neutral sentiment")






"""

https://www.datacamp.com/tutorial/serving-an-llm-application-as-an-api-endpoint-using-fastapi-in-python

# ollama run llama3
>>> give me the readability score of this  "Large Language Models (LLMs) are at the forefront of Generative AI, transforming how machines understand and generate huma-"l
... ike text. LLM Models like GPT, Claude, LLaMA, Mistral, etc. drive innovations in chatbots, content creation, and beyond, offering unprecedented capabilities in prc"e
... ssing and generating natural language." 
... 
>>> give me the readablity score  of this text  in json format "The current state-of-the-art models are created by OpenAI. While a single call to the OpenAI API is relt"
... ively cheap, costs add up at scale. This influenced how we designed our systems, we had to ensure that the cost was invariant to the number of visitors coming to or"
...  site. The upside is that the LLM space is extremely competitive and it is likely that this cost will be driven down significantly in the near future. Open source o"
... dels offer a much cheaper alternative, but as of writing this, they do not achieve comparable accuracy."
Here are the readability scores for the given text in JSON format:

```
{
  "Flesch-Kincaid Grade Level": {
    "score": 9.5,
    "range": "College-educated audience"
  },
  "Flesch-Kincaid Reading Ease": {
    "score": 70.1,
    "range": "Fairly easy to read for most readers"
  }
}
```

Note that the readability scores are based on the Flesch-Kincaid test, which estimates the grade level and reading ease of a text. The scores indicate that the 
text is suitable for a college-educated audience and is relatively easy to read.

"""