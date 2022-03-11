# IMPORTS
import re
import string
import time
import unidecode
from word2number import w2n
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# DOWNLOADS:
# nltk.download()
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


class PreProcessing(object):
    """
    SUMMARY:
    Class controlled by boolean variables, to perform text preprocessing based on the users preferences.
    The idea is to use this class and constantly update it as we come up with more methods of cleaning
    and preprocessing. Down the line, once we have our code setup via Dagshub/github, we can use this
    methods as a module which we can import from a separate .py file for easier data cleaning processes.

    CONTRIBUTORS:
    First_Name Last_Name - Task_2, Jr. Machine Learning Engineer
    Sara Hadou - Task_2, Jr. Machine Learning Engineer
    Aishwarya Sarkar - Task_2, Jr. Machine Learning Engineer

    NECESSARY LIBRARIES:
    import re
    import string
    import time
    import unidecode
    # import numpy as np
    # import pandas as pd
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from nltk.corpus import stopwords

    DOWNLOADS:
    nltk.download()
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    """

    def __init__(self, isTimer=False, isRemoveDigits=False, extended_report=False):
        """
        Constuctor, initializes the default flags for extra actions
        :param isTimer: bool - turn on/off the timer for pre-processing
        :param isRemoveDigits: bool - option to remove and not remove digit values
        :param extended_report: bool - option to save all removed/replaced values from unstructured data
        """
        self.extended_report = extended_report
        self.isRemoveDigits = isRemoveDigits
        self.isTimer = isTimer

    def remove_web_links(self, text="", weblink_save=False):
        """
        Removes weblinks from text. Open option to save removed links for further analysis
        :param text: str - unprocessed text
        :param weblink_save: bool - provide user with list of extracted weblinks
        :return: dict - without/with links
        """
        self.type_checker(text)
        result = {}
        pattern = r"(?i)\b((?:https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(" \
                  r"([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]" \
                  r"{};:'\".,<>?«»“”‘’]))"

        if weblink_save:
            weblink_list = re.findall(pattern, text)
            result['weblinks'] = [i[0] for i in weblink_list]

        result['text'] = re.sub(pattern, "", text)

        return result

    def remove_emojis(self, text="", emoji_save=False):
        """
        Removes emojis. Option to save the removed emoji if flag of emoji_save is True
        :param text: str - unprocessed text
        :param emoji_save: bool - provide the list of removed emojis
        :return: dict - with or without removed emojis with cleaned text
        """
        self.type_checker(text)
        result = {}
        pattern = re.compile(pattern="["
                                     u"\U0001F600-\U0001F64F"
                                     u"\U0001F300-\U0001F5FF"
                                     u"\U0001F680-\U0001F6FF"
                                     u"\U0001F1E0-\U0001F1FF"
                                     u"\U00002500-\U00002BEF"
                                     u"\U00002702-\U000027B0"
                                     u"\U00002702-\U000027B0"
                                     u"\U000024C2-\U0001F251"
                                     u"\U0001f926-\U0001f937"
                                     u"\U00010000-\U0010ffff"
                                     u"\u2640-\u2642"
                                     u"\u2600-\u2B55"
                                     u"\u200d"
                                     u"\u23cf"
                                     u"\u23e9"
                                     u"\u231a"
                                     u"\ufe0f"
                                     u"\u3030"
                                     "]+", flags=re.UNICODE)

        if emoji_save:
            emoji_list = re.findall(pattern, text)
            result['emoji_list'] = emoji_list

        result['text'] = pattern.sub('', text)

        return result

    def remove_spaces(self, text=""):
        """
        Removes extra spaces
        :param text: str - unprocessed text
        :return: str - without extra spaces
        """
        self.type_checker(text)
        result = {}

        text = re.sub(r'\n', "", text)
        result['text'] = re.sub(r' {2,}', " ", text)

        return result

    def remove_stopwords(self, text="", stopword_save=False):
        """
        Removes stopwords. Option to save the removed words if flag stopword_save is True
        :param text: str - unprocessed text
        :param stopword_save: bool - save stopwords into separate list
        :return: dict - cleaned text without stopwords with/without list of removed words
        """
        self.type_checker(text)
        result = {}

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())

        if stopword_save:
            stopwords_list = [w for w in words if w in stop_words]
            sentence = [w for w in words if w not in stop_words]
            result['stopwords'] = stopwords_list
        else:
            sentence = [w for w in words if w not in stop_words]
        result['text'] = " ".join(sentence)

        return result

    def lemmatize_text(self, text=""):
        """
        Applying lemmatization to raw text
        :param text: str - unprocessed text
        :return: str - in its based form
        """
        self.type_checker(text)
        result = {}

        lemmatizer = WordNetLemmatizer()
        wordnet_map = {'N': wordnet.NOUN,
                       'V': wordnet.VERB,
                       'J': wordnet.ADJ,
                       'R': wordnet.ADV}

        post_tagged_text = nltk.pos_tag(text.lower().split())
        text = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in post_tagged_text]
        result['text'] = " ".join(text)

        return result

    def lowercase_text(self, text=""):
        """
        Converting in lower cases
        :param text: str - unprocessed text
        :return: str - in lower cases
        """
        self.type_checker(text)
        result = {'text': text.lower()}

        return result

    def remove_punctuations(self, text=""):
        """
        Removing punctuation characters
        :param text: str - unprocessed text
        :return: str - text with no punctuations
        """
        self.type_checker(text)
        result = {}

        result['text'] = text.translate(str.maketrans('', '', string.punctuation))

        return result

    def remove_digits(self, text="", digits_save=False):
        """
        Removing numbers. Optin to save the removed digits if digits_flag is set to True
        :param text: str - unprocessed text
        :param text: bool - to save removed digits into separate list
        :return: dict - with no numbers or with/without removed digits
        """
        self.type_checker(text)
        result = {}

        if digits_save:
            text_list = text.split(' ')
            text_list = [''.join(x for x in i if x.isdigit()) for i in text_list]
            text_list = [i for i in text_list if len(i) > 0]
            result['digits'] = text_list

        text = text.replace(r'^\d+\.\s+', '')
        result['text'] = re.sub("[0-9'«»’‚“”„‹›‘]", '', text).strip()

        return result

    def remove_accented_str(self, text="", accented_str_save=False):
        """
        ASCII transliterations of Unicode text.  Replacing for supporting Unicode into string
        :param text: str - unstructured text consist of Unicode
        :param accented_str_save: list - option to save Unicode words to be replaced
        :return: dic - clean text with or without Unicode words
        """
        self.type_checker(text)
        result = {}
        unicode_words = []
        text = text.lower().split(" ")

        if accented_str_save:
            def save_unicode_words(word=""):
                unicode_words.append(word)

                return word

            text = [unidecode.unidecode(save_unicode_words(i)) if unidecode.unidecode(i) != i else i for i in text]
            result['unicode_words'] = unicode_words
        else:
            text = [unidecode.unidecode(i) for i in text]

        result['text'] = " ".join(text)

        return result

    def replace_contraction(self, text="", contraction_save=False):
        """
        Expanding English language contractions
        :param text: str - raw text with contraction words
        :param contraction_save: bool - ability to save the contraction words
        :return: dict - of clean text and with or without replaced contraction words
        """
        self.type_checker(text)
        result = {}
        text = text.lower().split(" ")

        # contractions mapped to the complete words to be used
        contraction_map = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how're": "how are",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have",
        }
        if contraction_save:
            contraction_word_list = []

            def help_func(contraction_word):
                contraction_word_list.append(contraction_word)

                return contraction_word

            text = [contraction_map[help_func(i)] if i in contraction_map.keys() else i for i in text]
            result['contraction_words'] = contraction_word_list
        else:
            text = [contraction_map[i] if i in contraction_map.keys() else i for i in text]
        result['text'] = " ".join(text)

        return result

    def remove_word_number(self, text="", word_number_save=False):
        """
        Removes words that represent numbers/digits
        :param text: str - raw text to be processes
        :param word_number_save: bool - option to save word-number as separate list
        :return: dict - will consist str with no word-number and with/without list of removed word-numbers
        """
        self.type_checker(text)
        result = {}
        word_to_num_list = []
        text = text.lower().split(" ")

        def check_conversion(word=""):
            """
            Helpful function to check if the word can be converted into number/digit
            :param word: str - text that could be used to covert into digit/nunmber
            :return: str - empty string if the word is converted otherwise the original word
            """
            try:
                word_converted = w2n.word_to_num(word)
                if word_number_save:
                    word_to_num_list.append(word_converted)

                return True
            except ValueError:

                return False

        if word_number_save:
            result['word_to_number'] = word_to_num_list

        text = [i for i in text if not check_conversion(i)]
        result['text'] = " ".join(text)

        return result

    def type_checker(self, type_input):
        if not isinstance(type_input, str):
            raise TypeError("only string accepted as parameter.")

        return True

    def apply_all_cleaning_steps(self, text=""):
        """
        Applying all cleaning steps to pre-process raw data
        :param text: str - unprocessed text
        :return: str - cleaned data
        """
        self.type_checker(text)
        start_time = time.time()
        result = {}

        if self.extended_report:
            text = self.lowercase_text(text)
            text = self.replace_contraction(text['text'], contraction_save=True)
            result['contraction_words'] = text['contraction_words']

            text = self.remove_accented_str(text['text'], accented_str_save=True)
            result['unicode_words'] = text['unicode_words']

            text = self.remove_web_links(text['text'], weblink_save=True)
            result['weblinks'] = text['weblinks']

            text = self.remove_emojis(text['text'], emoji_save=True)
            result['emoji_list'] = text['emoji_list']

            if self.isRemoveDigits:
                text = self.remove_digits(text['text'], digits_save=True)
                result['digits'] = text['digits']

            text = self.remove_punctuations(text['text'])

            if self.isRemoveDigits:
                text = self.remove_word_number(text['text'], word_number_save=True)
                result['word_to_number'] = text['word_to_number']

            text = self.remove_stopwords(text['text'], stopword_save=True)
            result['stopwords'] = text['stopwords']

            text = self.remove_spaces(text['text'])
            text = self.lemmatize_text(text['text'])
        else:
            text = self.lowercase_text(text)
            text = self.replace_contraction(text['text'])
            text = self.remove_accented_str(text['text'])
            text = self.remove_web_links(text['text'])
            text = self.remove_emojis(text['text'])

            if self.isRemoveDigits:
                text = self.remove_digits(text['text'])
            text = self.remove_punctuations(text['text'])

            if self.isRemoveDigits:
                text = self.remove_word_number(text['text'])

            text = self.remove_stopwords(text['text'])
            text = self.remove_spaces(text['text'])
            text = self.lemmatize_text(text['text'])

        result['text'] = text['text']
        end_time = time.time()

        if self.isTimer:
            print(f"data pre-process time: {round(end_time - start_time, 2)} sec\n")

        return result
