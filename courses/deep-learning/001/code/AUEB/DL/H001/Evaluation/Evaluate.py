import pandas as pd

class Evaluate(object):

    """
    """
    def __init__(self):
        self.__metrics = None

    """
    """
    def evaluate(self, histories, X_test, y_test):
        self.__metrics = {}
        
        for i, history_label in enumerate(histories):
            self.__metrics[history_label] = histories[history_label].model.evaluate(X_test, y_test)
            
        return self

    """
    """
    def to_df(self):
        return pd.DataFrame.from_dict(
            self.__metrics, 
            orient = "index",
            columns = ["test_loss", "test_accuracy"]
        )