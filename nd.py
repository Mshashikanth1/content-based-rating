import nltk
import ssl
import spacy.cli

# Download and install the English model
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

spacy.cli.download("en_core_web_sm")

