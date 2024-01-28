# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse, string, math, nltk, csv
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca21cml" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    # read files using pandas
    training = pd.read_csv(inputs.training, sep='\t')
    dev = pd.read_csv(inputs.dev, sep='\t')
    test = pd.read_csv(inputs.test, sep='\t')
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    
    def check_nltk_resources():
        """
        Checks if the necessary NLTK resources have been downloaded.
        If not, download the required resources.
    
        Returns:
        - downloaded (bool): True if resources are available, False if not.
        """
        try:
            stopwords.words('english')
            WordNetLemmatizer().lemmatize('test')
            return
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('stopwords')
            nltk.download('wordnet')
            return
    
    def format_data(data):
        """
       Formats the input data into a suitable format for sentiment analysis.
    
       Parameters:
       - data (DataFrame): A pandas DataFrame containing the input data, which may include a 'Sentiment' column
         for training and development data.
    
       Returns:
       - data_list (list): A list containing formatted data. For training and development data, each element
         in the list is a tuple (phrase, sentiment), and for test data, each element is a string representing the phrase.
         
       """
        data_list = []
    
        if 'Sentiment' in data.columns: # format training and dev data
            for _, row in data.iterrows():
                sentiment = row['Sentiment']
    
                # descale sentiment
                if number_classes == 3:
                    if sentiment == 1:
                        sentiment = 0
                    elif sentiment == 2:
                        sentiment = 1
                    elif sentiment >= 3:
                        sentiment = 2
    
                data_list.append((row['Phrase'], sentiment))
        else: # format test data
            for _, row in data.iterrows():
                data_list.append(row['Phrase'])
    
        return data_list
    
    def preprocess(data):
        """
        Preprocesses input data for sentiment analysis based on the type of features to use.
    
        Parameters:
        - data (list): A list of tuples or strings representing input data. If tuples, each tuple should contain a
          phrase and its sentiment label. If strings, it represents the test data.
    
        Returns:
        - preprocessed_data (list): A list containing preprocessed data. For training and development data,
          each element in the list is a tuple (preprocessed_phrase, sentiment), and for test data, each element
          is a preprocessed list of words representing the phrase.
    
        """
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        def preprocess_phrase(phrase):
            # separate linked words and remove punctuations
            if features == 'all_words':
                phrase = phrase.replace('-', ' ').translate(str.maketrans("", "", string.punctuation))

            # tokenization
            phrase = phrase.split()

            # lowercase
            phrase = [word.lower() for word in phrase]

            # lemmatization
            phrase = [lemmatizer.lemmatize(word) for word in phrase]

            # stoplisting
            phrase = [word for word in phrase if word not in stop_words]
            
            return phrase
        
        for i, item in enumerate(data):
            if isinstance(item, tuple): # preprocess training and dev data
                phrase, sentiment = item
                data[i] = (preprocess_phrase(phrase), sentiment)
    
            else: # preprocess test data
                phrase = item
                data[i] = preprocess_phrase(phrase)
           
    def compute_tfidf(data):
        """
        Computes TF-IDF values for each feature in the input data across different sentiment classes.
    
        Parameters:
        - data (list): A list of tuples, each containing a preprocessed phrase and its corresponding sentiment label.
    
        Returns:
        dict: A dictionary where keys are distinct features in the dataset, and values are lists
              representing the sum of TF-IDF values across documents, with indices corresponding to sentiment classes.
    
        """
        df = {}
        idf = {}
        tfidf = {}
    
        # compute document freq for each feature
        for phrase, _ in data:
            for feat in set(phrase):
                df[feat] = df.get(feat, 0) + 1

        # compute idf for each feature
        data_len = len(data)
        for feat in df.keys():
            idf[feat] = math.log(data_len / df[feat])
    
        # compute tf-idf for each feature in each sentiment class
        for phrase, sentiment in data:
            for feat in phrase:
                if feat not in tfidf:
                    tfidf[feat] = [0] * number_classes # initialize tfidf for each class
                tfidf[feat][sentiment] += (df[feat] / len(phrase)) * idf[feat]
        
        return tfidf
    
    def bayes_classifier(data):
        """
        Trains a Naive Bayes classifier on the provided data for sentiment analysis based on the type of 
        features to use for classification.
    
        Parameters:
        - data (list): A list of tuples representing training data. Each tuple contains a preprocessed phrase
          and its corresponding sentiment label.

        Returns:
        - prior_prob (list): A list representing the prior probabilities for each sentiment class.
        - likelihoods (list of dictionaries): A list of dictionaries containing likelihood probabilities
          for each feature in each sentiment class.

        """
        prior_prob = [0] * number_classes
        likelihoods = [{} for _ in range(number_classes)]
        
        data_len = len(data)
        
        # compute prior prob for each sentiment class
        for phrase, sentiment in data:
            prior_prob[sentiment] += 1
        prior_prob = [x / data_len for x in prior_prob]
        
        bag = [{} for _ in range(number_classes)] 
        distinct_feats = set()
        
        for phrase, sentiment in data:
            for feat in phrase:
                # calculate bag of words for each sentiment class
                bag[sentiment][feat] = bag[sentiment].get(feat, 0) + 1
                    
                # get all distinct features in the dataset     
                distinct_feats.add(feat)
        
        # total number of distinct features in the dataset
        total_distinct_feats = len(distinct_feats)
        
        # calculate total number of features in each sentiment class
        total_feats = [sum(bag[sentiment].values()) for sentiment in range(number_classes)]
        
        # calculate likelihoods for each feature in each sentiment class:
        if features == 'all_words': # using all words as features
            for feat in distinct_feats:
                for sentiment in range(number_classes):
                    numerator = bag[sentiment].get(feat, 0) + 1 # add 1 for smoothing
                    denominator = total_feats[sentiment] + total_distinct_feats # smoothing
                    likelihoods[sentiment][feat] = numerator / denominator
        else: # using feature extraction
            aggregated_tfidf = {}
            aggregated_tfidf_class = [0] * number_classes
            
            # compute tf-idf of training data
            tfidf = compute_tfidf(data)
            
            for feat, tfidf_values in tfidf.items():
                for i in range(number_classes):
                    aggregated_tfidf_class[i] += tfidf_values[i] # sum of tf-idf of features in each sentiment class
            
            # compute likelihoods for each feature in each sentiment class based on tf-idf
            for feat in tfidf.keys():
                for sentiment in range(number_classes):
                    numerator = (bag[sentiment].get(feat, 0) + 1) * (tfidf[feat][sentiment] + 1) # add 1 for smoothing
                    denominator = (total_feats[sentiment] + total_distinct_feats) * (aggregated_tfidf_class[sentiment]) # smoothing
                    likelihoods[sentiment][feat] = numerator / denominator
        
        return prior_prob, likelihoods
    
    def predict_sentiment(prior_prob, likelihoods, data):
        """
        Predicts sentiment labels for a given set of phrases using a Naive Bayes classifier.
    
        Parameters:
        - prior_prob (list): A list representing the prior probabilities for each sentiment class.
        - likelihoods (list of dictionaries): A list of dictionaries containing likelihood probabilities
          for each feature in each sentiment class.
        - data (list of tuples or lists): The input data to be classified. Each element in the list can be a
          tuple (for development data) or a list (for test data), where the first element is the phrase to be
          classified, and the second element (if present) is the true sentiment label.
    
        Returns:
        - results (list): A list containing the predicted sentiment labels for each input phrase.

        """
        results = []
    
        for item in data:
            if isinstance(item, tuple): # predict dev data
                phrase, _ = item
            else: # predict test data
                phrase = item
            posterior_prob = prior_prob.copy() # copy prior prob to initiate posterior prob
            for sentiment in range(number_classes):
                for feat in phrase:
                    likelihood = likelihoods[sentiment].get(feat, 1) # default to 1 if feature does not exist for smoothing
                    posterior_prob[sentiment] *= likelihood
    
            predicted = posterior_prob.index(max(posterior_prob)) # get sentiment class with highest posterior prob
            results.append(predicted)
    
        return results
    
    def macro_f1_score(data, results):
        """
        Computes the macro-F1 score and confusion matrix for a classifier's predictions.
    
        Parameters:
        - data (list): A list of tuples, each containing a phrase and its true sentiment label.
        - results (list): A list of predicted sentiment labels corresponding to the phrases in the 'data' parameter.
    
        Returns:
        tuple: A tuple containing the macro-F1 score and the confusion matrix.
    
        """
        # compute confusion matrix
        confusion = [[0 for _ in range(number_classes)] for _ in range(number_classes)] 
        for i, (_, sentiment) in enumerate(data):
            confusion[sentiment][results[i]] += 1
        
        # compute f1 score for each sentiment class
        f1_score = 0
        for i in range(number_classes):  
            tp = confusion[i][i]
            fp = sum(confusion[j][i] for j in range(number_classes) if j != i)
            fn = sum(confusion[i][j] for j in range(number_classes) if j != i)
            tn = sum(confusion[j][k] for j in range(number_classes) for k in range(number_classes) if j != i and k != i) 

            f1_score += (2 * tp) / (2 * tp + fp + fn)
        
        # compute macro f1 score
        f1_score = f1_score / number_classes
    
        return f1_score, confusion
    
    def visualise_confusion_matrix(confusion):
        """
        Visualizes the confusion matrix using a heatmap.
    
        Parameters:
        - confusion (list): The confusion matrix, a 2D list representing the counts of true and predicted labels.
    
        Returns:
        None: Displays the heatmap plot.
    
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[x for x in range(number_classes)],
                    yticklabels=[x for x in range(number_classes)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def save_results(data, results, datastr):
        """
        Outputs the predicted sentiment labels to a TSV file.
    
        Parameters:
        - data (DataFrame): The input DataFrame containing 'SentenceId' column.
        - results (list): A list of predicted sentiment labels corresponding to the 'data' parameter.
        - datastr (str): Name for the file to be saved.
    
        Returns:
        None: Writes the TSV file with the predicted results.
        
        """
        file_name = f'{datastr}_predictions_{number_classes}classes_{USER_ID}.tsv'
        
        # write to tsv file
        with open(file_name, 'w', newline='', encoding='utf-8') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t')
            
            # write header
            tsv_writer.writerow(['SentenceId', 'Sentiment'])
            
            # write each row
            for i, row in data.iterrows():
                tsv_writer.writerow([row['SentenceId'], results[i]])
                
        print(f'Results saved for {file_name}!')
    
    """
    Main code
    """
    # download required nltk resources if missing
    check_nltk_resources()
    
    # format data and descale sentiment if specified
    training_list = format_data(training)
    dev_list = format_data(dev)
    test_list = format_data(test)
        
    # preprocess data
    preprocess(training_list)
    preprocess(dev_list)
    preprocess(test_list)
    
    # train model
    prior_prob, likelihoods = bayes_classifier(training_list)
        
    # predict sentiment
    dev_results = predict_sentiment(prior_prob, likelihoods, dev_list) # dev
    test_results = predict_sentiment(prior_prob, likelihoods, test_list) # test
    
    # compute macro-f1 and confusion matrix 
    f1_score, confusion = macro_f1_score(dev_list, dev_results)
    
    # visualise confusion matrix if specified
    if confusion_matrix:
        visualise_confusion_matrix(confusion)
        
    # output tsv files if specified
    if output_files:
        save_results(dev, dev_results, 'dev')
        save_results(test, test_results, 'test')

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()