import pandas as pd
import numpy as np
import math


class SMSDataPreProcessor():

    def __init__(self, path, sep, cols, test_size):
        self.path = path
        self.seperator = sep
        self.columns = cols
        self.test_size = test_size
        self.extract()
        # print(self.rawData)
        self.preprocess()
        # print(self.preprocessDF)
        self.train, self.test = self.sample()

    def extract(self):
        self.rawData = pd.read_table(
            self.path, sep=self.seperator, names=self.columns)

    def checkASCII(self, c):
        ascii = ord(c)
        if (ascii >= 97 and ascii <= 122):
            return True
        elif (ascii >= 48 and ascii <= 57):
            return True
        elif (ascii == 36):
            return True
        else:
            return False

    def cleanText(self, rawText):
        
        stop_words = [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 
            'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'they', 
            'their', 'this', 'these', 'those', 'or', 'as', 'but', 'not', 'if', 'then', 'else', 
            'when', 'where', 'which', 'who', 'whom', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 'can', 'should', 'now', 'i', 'im', 'u', 'you', 'me', 'my', 'ur', ''
        ]


        wordList = rawText.lower().split()
        cleanWords = []
        for word in wordList:
            newWord = ""
            flag = True
            for char in word:
                if self.checkASCII(char):
                    newWord = newWord+char
                # elif len(newWord)>0 and flag:
                #     newWord=newWord+" "
                #     flag=False
            if (len(newWord)>2 and newWord not in stop_words):
                cleanWords.append(newWord)
        cleanedText = " ".join([cleanWord for cleanWord in cleanWords])
        return cleanedText

    def preprocess(self):
        preprocessDF = self.rawData
        preprocessDF["cleanText"] = preprocessDF.apply(
            lambda row: self.cleanText(row["text"]), axis=1)
        preprocessDF["label"] = preprocessDF["label"].map(
            {'ham': 0, 'spam': 1})
        self.preprocessDF = preprocessDF.drop("text", axis=1)

    def sample(self):
        np.random.seed(133)
        train_size = math.ceil((1-self.test_size)*len(self.preprocessDF))
        permuted_indices = np.random.permutation(len(self.preprocessDF))
        df_train = self.preprocessDF.iloc[permuted_indices[0:train_size]]
        df_test = self.preprocessDF.iloc[permuted_indices[train_size:]]
        return df_train, df_test


class SMSDataTransformer():

    def __init__(self, k):
        self.k = k

    def trainPipeline(self, data, mode):
        self.data = data
        if mode=="count":
            self.bagOfWords()
        elif mode=="tf-idf":
            self.compute_tf_idf()
        self.createBinaryVector()
        return self.vocab, self.binary_encoded_data, self.feature_list

    def testPipeline(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.createBinaryVector()
        return self.binary_encoded_data
    

    def compute_tf_idf(self):
        docs=self.data["cleanText"].to_numpy().flatten()
        tf_idf = {}
        N = len(docs)

        # Compute term frequency (tf) for each document
        for doc in docs:
            for word in doc.split():
                tf = doc.count(word) / len(doc)

                # Compute inverse document frequency (idf)
                df = sum([1 for d in docs if word in d.split()])
                idf = math.log(N / df)

                # Compute tf-idf score
                tf_idf_score = tf * idf

                # Add tf-idf score to dictionary
                if word not in tf_idf:
                    tf_idf[word] = tf_idf_score
                else:
                    tf_idf[word] += tf_idf_score

        self.vocab = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:self.k]
        print(self.vocab)


    def bagOfWords(self):
        flat_str = " ".join(self.data["cleanText"].to_numpy().flatten())
        words = flat_str.split()
        counts = dict()
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        self.vocab = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
            0:self.k]

    def createBinaryVector(self):
        binary_encoded_data = self.data.copy()
        feature_list = []
        for tup in self.vocab:
            feature_list.append(tup[0])
            bool_series = binary_encoded_data["cleanText"].str.contains(
                fr'\b{tup[0]}\b', regex=True, case=False)
            binary_encoded_data[str(tup[0])] = [int(val)
                                                for val in bool_series]
        self.binary_encoded_data = binary_encoded_data
        self.feature_list = feature_list
        print(feature_list)
        # print(binary_encoded_data)


class NaiveBayesModel():
    def __init__(self, features):
        self.features = features

    def train(self, data):
        self.data = data
        prior = self.calculatePrior()
        likelihoods = self.calculateLikelihood()
        return prior, likelihoods

    def predict(self, prior, likelihoods, data):
        return self.calculatePosterior(prior, likelihoods, data)

    def calculatePrior(self):
        classes, counts = np.unique(self.data["label"], return_counts=True)
        class_prob = dict()
        for i, cls in enumerate(classes):
            class_prob[cls] = counts[i] / len(self.data["label"])
        print(class_prob)
        return class_prob

    def calculateLikelihood(self):
        classes = np.unique(self.data["label"])
        feature_prob = dict(dict())
        for cls in classes:
            class_rows = self.data[self.data["label"] == cls]
            for feature in self.features:
                if cls not in feature_prob:
                    feature_prob[cls] = dict()
                if feature not in feature_prob[cls]:
                    feature_prob[cls][feature] = (
                        class_rows[feature].sum() + 1) / (len(class_rows) + 2)
        print(feature_prob)
        return feature_prob

    def calculatePosterior(self, prior, likelihoods, data):
        y_pred = []
        for _, row in data.iterrows():
            posterior_prob = dict()
            for cls in prior.keys():
                posterior_prob[cls] = prior[cls]
                for feature in self.features:
                    posterior_prob[cls] *= likelihoods[cls][feature] if row[feature] == 1 else 1 - \
                        likelihoods[cls][feature]
            y_pred.append(max(posterior_prob, key=posterior_prob.get))
        predictions = data.copy()
        predictions["predicted_label"] = y_pred
        # print(predictions)
        return predictions


class Metrics():

    def calculate_metrics(y_true, y_pred):

        accuracy = (y_true == y_pred).sum()/len(y_true)

        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))

        confusion_matrix = pd.DataFrame({
            'Actual Positive': [true_positives, false_negatives],
            'Actual Negative': [false_positives, true_negatives]
        }, index=['Predicted Positive', 'Predicted Negative'])

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return accuracy, precision, recall, f1_score, confusion_matrix


def main():
    sms = SMSDataPreProcessor(path="smsspamcollection/SMSSpamCollection",
                              sep="\t", cols=["label", "text"], test_size=0.2)
    train_transformer = SMSDataTransformer(k=15)
    vocab, binary_encoded_train_data, features = train_transformer.trainPipeline(
        data=sms.train, mode="tf-idf")
    nb = NaiveBayesModel(features=features)
    prior, likelihoods = nb.train(binary_encoded_train_data)
    spam_prediction_train_data = nb.predict(
        prior=prior, likelihoods=likelihoods, data=binary_encoded_train_data)
    test_transformer = SMSDataTransformer(k=15)
    binary_encoded_test_data = test_transformer.testPipeline(
        data=sms.test, vocab=vocab)
    spam_prediction_test_data = nb.predict(
        prior=prior, likelihoods=likelihoods, data=binary_encoded_test_data)

    print("Train Metrics")

    accuracy, p, r, f1, cm = Metrics.calculate_metrics(
        y_true=spam_prediction_train_data["label"], y_pred=spam_prediction_train_data["predicted_label"])
    print(accuracy, p, r, f1)
    print(cm)

    print("Test Metrics")

    accuracy, p, r, f1, cm = Metrics.calculate_metrics(
        y_true=spam_prediction_test_data["label"], y_pred=spam_prediction_test_data["predicted_label"])
    print(accuracy, p, r, f1)
    print(cm)


if __name__ == "__main__":
    main()
