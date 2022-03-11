import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras


class TweetsSentimentAnalysis(object):
    """
    This class represents the instance of DDe
    """

    def __init__(self, built_model=None):
        self.model_obj = built_model

    def load_model(self, path_to_model=""):
        """
        Loading the model from the provided path and saving as instance of this class
        :param path_to_model: str - relative path to the existing model from the root folder
        :return: bool - true if there were no issues of loading the model otherwise false
        """
        if self.model_obj is not None:
            user_input = input("this class object already has loaded model. would you like to override it?\n"
                               "accepted values yes/no: ")
            while True:
                if user_input.strip().lower() not in ['yes', 'no']:
                    user_input = input('error. provided value is not accepted. please type only: "yes" or "no"')
                    continue
                break

            if user_input.strip().lower() == 'no':
                return True
        try:
            self.model_obj = keras.models.load_model(path_to_model)

            return True
        except:
            return False

    def get_prediction(self, data_set=""):
        """
        Estimates the value based on the provided values
        :param data_set: dict - set of features to be used for model prediction
        :return: float - the predicted value based on the provided information
        """
        results = {}
        _max_sequence_length = 30
        token_instance = pickle.load(open('tokenizer.pickle', 'rb'))

        data_set_list = [data_set]
        seq = token_instance.texts_to_sequences([data_set_list])
        padded = pad_sequences(seq, maxlen=_max_sequence_length)

        y_pred = self.model_obj.predict(padded)
        y_pred = ['bad' if x[0] <= 0.5 else 'good' for x in y_pred]
        results['prediction'] = y_pred

        return results
