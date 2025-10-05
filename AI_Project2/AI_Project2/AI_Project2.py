from cgi import test
from os import remove
from matplotlib.pylab import rand
import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from statistics import mode
import numpy as np
import math
from functools import partial

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
        


class ID3:
    def __init__(self, features):
        self.tree = None
        self.features = features
    
    def fit(self, x, y):
        '''
        creates the tree
        '''
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common)
        return self.tree
    
    def create_tree(self, x_train, y_train, features, category):
        
        # check empty data
        if len(x_train) == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
        
        # check all examples belonging in one category
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
        
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        
        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))
        
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, 
                                           category=m)  # go left for X = 1
        
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m)  # go right for X = 0
        
        return root


    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    

        

    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x:  # for every example 
            tmp = self.tree  # begin at root
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)


def calculate_information_gain(df, m=10):
    """
    Calculates the information gain of each word and returns
    the top m words with highest information gain.
    """
    def entropy(probabilities):
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].values
    words = np.array(vectorizer.get_feature_names_out())
    
    # Entropy of the class distribution
    H_C = entropy(np.bincount(y) / len(y))
    information_gain = {}
    
    # Calculate the information gain for each word
    for i, word in enumerate(words):
        word_presence = X[:, i].toarray().flatten()
        P_X = np.bincount(word_presence) / len(word_presence)
        H_C_given_X = sum((P_X[j] * entropy(np.bincount(y[word_presence == j]) / len(y[word_presence == j]))
                           for j in range(len(P_X)) if len(y[word_presence == j]) > 0))
        IG = H_C - H_C_given_X
        information_gain[word] = IG
    
    top_m_words = sorted(information_gain, key=information_gain.get, reverse=True)[:m]
    return top_m_words

def extract_frequent_words(df, n=10, k=100):
    """
    Extracts the n most frequent and k least frequent words from the review texts.
    """
    all_text = ' '.join(df['review'].astype(str))
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    word_counts = Counter(words)
    most_common_words = [word for word, _ in word_counts.most_common(n)]
    least_common_words = [word for word, _ in word_counts.most_common()[-k:]]
    return most_common_words, least_common_words

def remove_words(df, words_to_remove):
    """
    Removes all occurrences of the words in words_to_remove from the review column.
    """
    pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
    df['review'] = df['review'].astype(str).apply(
        lambda text: re.sub(pattern, '', text, flags=re.IGNORECASE)
    ).str.replace('\s+', ' ', regex=True).str.strip()
    return df

def keep_top_m_words(df, top_m_words):
    """
    Keeps only the words that are in the top_m_words list in each review.
    """
    pattern = r'\b(' + '|'.join(map(re.escape, top_m_words)) + r')\b'
    df['review'] = df['review'].astype(str).apply(
        lambda text: ' '.join(re.findall(pattern, text.lower()))
    )
    return df

def calculateProbabilities(df):
    X = df.drop(columns=['sentiment']).values
    y = df['sentiment'].values    

    N = len(y)                     # total number of training samples
    N_neg = np.sum(y == 0)         # number of negative samples
    N_pos = np.sum(y == 1)         # number of positive samples
    m = X.shape[1]   
    
    pC0 = N_neg / N
    pC1 = N_pos / N

    # For class 0
    X_neg = X[y == 0]
    count_ones_neg = X_neg.sum(axis=0)  
    pWord1Given0 = (count_ones_neg + 1.0) / (N_neg + 2.0)

    # For class 1
    X_pos = X[y == 1]
    count_ones_pos = X_pos.sum(axis=0)
    pWord1Given1 = (count_ones_pos + 1.0) / (N_pos + 2.0)

    return {
        'pC0': pC0,
        'pC1': pC1,
        'pWord1Given0': pWord1Given0,
        'pWord1Given1': pWord1Given1
    }

def classify_naive_bayes(model, df):
    """
    Classifies each row in df using the provided Naive Bayes model parameters.
    """
    pC0 = model['pC0']
    pC1 = model['pC1']
    pWord1Given0 = model['pWord1Given0']
    pWord1Given1 = model['pWord1Given1']

    X = df.drop(columns=['sentiment']).values  # shape: (num_samples, m)
    pWord0Given0 = 1 - pWord1Given0
    pWord0Given1 = 1 - pWord1Given1

    predictions = []
    for x in X:
        prob0 = pC0 * np.prod(x * pWord1Given0 + (1 - x) * pWord0Given0)
        prob1 = pC1 * np.prod(x * pWord1Given1 + (1 - x) * pWord0Given1)
        prediction = 1 if prob1 > prob0 else 0
        predictions.append(prediction)

    return pd.Series(predictions)

def print_scores(y_true, y_pred):
    """
    Computes and prints F1, Recall, and Precision scores.
    """
    f_score_macro = f1_score(y_true, y_pred, average='macro')
    f_score_micro = f1_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    f_score_pos = f1_score(y_true, y_pred, pos_label=1)
    recall_pos = recall_score(y_true, y_pred, pos_label=1)
    precision_pos = precision_score(y_true, y_pred, pos_label=1)
    
    print("=== Evaluation Scores ===")
    print("F1 Score (macro):     ", f_score_macro)
    print("F1 Score (micro):     ", f_score_micro)
    print("Recall (macro):       ", recall_macro)
    print("Recall (micro):       ", recall_micro)
    print("Precision (macro):    ", precision_macro)
    print("Precision (micro):    ", precision_micro)
    print("F1 Score (positive):  ", f_score_pos)
    print("Recall (positive):    ", recall_pos)
    print("Precision (positive): ", precision_pos)
    print("==========================")
    
    return f_score_pos, recall_pos, precision_pos

def execute_model(model_builder, predictor, training_data, test_data, model_name="Model"):
    """
    Builds the model, evaluates it on both test and training data,
    prints the scores, and returns a tuple of evaluation metrics.
    """
    model = model_builder(training_data)
    
    print(f">>> {model_name} Evaluation on TEST Data:")
    y_pred_test = predictor(model, test_data)
    test_f1, test_recall, test_precision = print_scores(test_data["sentiment"], y_pred_test)
    print("------------------------")
    
    print(f">>> {model_name} Evaluation on TRAINING Data:")
    y_pred_train = predictor(model, training_data)
    train_f1, train_recall, train_precision = print_scores(training_data["sentiment"], y_pred_train)
    print("------------------------")
    
    return (test_f1, test_recall, test_precision, train_f1, train_recall, train_precision)

def run_sklearn_bernoulli(training_data, test_data):
    """
    Trains a BernoulliNB classifier using the provided training_data,
    predicts on both test and training data, and prints the scores.
    """
    bernoulli_clf = BernoulliNB()
    bernoulli_clf.fit(training_data.drop(columns=['sentiment']).values, training_data["sentiment"])
    y_pred_bernoulli = bernoulli_clf.predict(test_data.drop(columns=['sentiment']).values)
    print("=== BernoulliNB Test Data Scores ===")
    print_scores(test_data["sentiment"], y_pred_bernoulli)
    print("------------------------")
    y_pred_bernoulli = bernoulli_clf.predict(training_data.drop(columns=['sentiment']).values)
    print("=== BernoulliNB Training Data Scores ===")
    print_scores(training_data["sentiment"], y_pred_bernoulli)

def create_random_forest(training_data, n):
    forest = []
    for i in range(n):
        sample = training_data.sample(n=len(training_data), replace=True)
        
        feature_columns = [col for col in training_data.columns if col != 'sentiment']
        
        # Randomly select 70 features from the available features
        print(f"Creating tree {i + 1}")
        selected_features = np.random.choice(feature_columns, size=70, replace=False)
        
        x_train = sample[selected_features].values
        y_train = sample['sentiment'].to_numpy()
        
        # Create an ID3 tree using the selected feature names.
        tree = ID3(selected_features)
        
        # Fit the tree with the prepared training data.
        tree.fit(x_train, y_train)
        forest.append(tree)
    
    return forest

def majority_vote_predict(forest, test_data):
    all_tree_predictions = []
    for tree in forest:
        x_test_subset = test_data[tree.features].values
        preds = tree.predict(x_test_subset)
        all_tree_predictions.append(preds)
    
    all_tree_predictions = np.array(all_tree_predictions)
    
    n_trees, n_examples = all_tree_predictions.shape
    final_predictions = []
    
    # For each test example, compute the majority vote.
    for j in range(n_examples):
        votes = all_tree_predictions[:, j]
        counts = np.bincount(votes)
        final_predictions.append(np.argmax(counts))
    
    # Return as a pandas Series
    return pd.Series(final_predictions, index=test_data.index)

def run_sklearn_randomforest(training_data, test_data):
    rf_clf = RandomForestClassifier(n_estimators = 11, max_features = 60, criterion = "entropy")
    x_train = training_data.drop(columns=['sentiment']).values
    y_train = training_data["sentiment"].values
    rf_clf.fit(x_train, y_train)
    y_pred_rf = rf_clf.predict(test_data.drop(columns=['sentiment']).values)
    print("=== Random Forest Test Data Scores ===")
    print_scores(test_data["sentiment"], y_pred_rf)
    print("------------------------")
    y_pred_rf = rf_clf.predict(training_data.drop(columns=['sentiment']).values)
    print("=== Random Forest Training Data Scores ===")
    print_scores(training_data["sentiment"], y_pred_rf)

def fit_adaboost(training_data, M=5):
    """
    Train AdaBoost with M stumps
    """
    # Separate features from labels
    X = training_data.iloc[:, :-1].values  # all but the last column
    y_0_or_1 = training_data.iloc[:, -1].values  # last column (sentiment)
    N, num_features = X.shape
    
    # Convert labels from {0,1} to {-1,+1} for AdaBoost math
    y = 2 * y_0_or_1 - 1  # now y in {-1, +1}
    
    # Initialize sample weights: D_i = 1/N
    D = np.ones(N) / N
    
    stumps = []  # will store the feature indices used by each stump
    alphas = []  # will store alpha_t for each stump
    
    for t in range(M):
        # Stump uses the t-th column => "best" feature for iteration t
        # Predictions h_t(x) = +1 if feature=1 else -1
        stump_predictions = np.where(X[:, t] == 1, +1, -1)
        
        # Weighted error e_t
        mismatch = (stump_predictions != y)  # boolean array
        e_t = np.sum(D[mismatch])
        
        # Compute alpha_t = 0.5 * ln((1 - e_t) / e_t)
        eps = 1e-10
        e_t = max(min(e_t, 1 - eps), eps)
        alpha_t = 0.5 * np.log((1 - e_t) / e_t)
        
        # Update D_i = D_i * exp(-alpha_t * y_i * h_t(x_i))
        D *= np.exp(-alpha_t * y * stump_predictions)
        D /= np.sum(D)  # normalize so sum(D)=1
        
        stumps.append(t)
        alphas.append(alpha_t)
    
    return stumps, alphas

def predict_adaboost(df, stumps, alphas):
    X = df.values  # shape (N, num_features)
    N = len(X)
    
    # Sum alpha_t * h_t(x)
    total = np.zeros(N)
    for t, alpha_t in zip(stumps, alphas):
        stump_preds = np.where(X[:, t] == 1, +1, -1)
        total += alpha_t * stump_preds
    
    # sign(...) in {-1,+1}, map -1->0, +1->1
    y_pred_pm = np.sign(total)   # sign(0)=0 => treat as +1 or -1
    y_pred_01 = (y_pred_pm + 1) // 2
    return y_pred_01

def build_adaboost(training_data, M):
    stumps, alphas = fit_adaboost(training_data, M)
    return {
        "stumps": stumps,
        "alphas": alphas
    }

def classify_adaboost(model, df):
    """
    Classifies df using a model dict from build_adaboost.
    """
    feature_df = df.drop(columns=['sentiment'], errors='ignore')
    y_pred_01 = predict_adaboost(feature_df, model["stumps"], model["alphas"])
    return pd.Series(y_pred_01, index=df.index)

def run_sklearn_adaboost(training_data, test_data):
    adaboost_clf = AdaBoostClassifier(n_estimators = 100)
    x_train = training_data.drop(columns=['sentiment']).values
    y_train = training_data["sentiment"].values
    adaboost_clf.fit(x_train, y_train)
    y_pred_rf = adaboost_clf.predict(test_data.drop(columns=['sentiment']).values)
    print("=== Adaboost Test Data Scores ===")
    print_scores(test_data["sentiment"], y_pred_rf)
    print("------------------------")
    y_pred_rf = adaboost_clf.predict(training_data.drop(columns=['sentiment']).values)
    print("=== Adaboost Training Data Scores ===")
    print_scores(training_data["sentiment"], y_pred_rf)

""" Preprocessing:
# Read the full dataset
df = pd.read_csv('movie_data.csv')

# Split the dataset into 50% training and 50% test data
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# --- Preprocessing on training data ---
# Remove the 50 most frequent and 500 least frequent words
most_common_words, least_common_words = extract_frequent_words(train_df, n=50, k=500)
train_df = remove_words(train_df, most_common_words + least_common_words)

# Calculate the top 5000 words based on information gain from the training data
vocab = calculate_information_gain(train_df, m=5000)

# Restrict training reviews to only include these top 5000 words
train_df = keep_top_m_words(train_df, vocab)

# --- Apply similar processing to test data for consistency ---
test_df = remove_words(test_df, most_common_words + least_common_words)
test_df = keep_top_m_words(test_df, vocab)

# --- Create binary feature vectors using the fixed vocabulary ---
vectorizer = CountVectorizer(vocabulary=vocab, binary=True)

# Transform the review texts into a binary matrix
X_train = vectorizer.transform(train_df['review'])
X_test = vectorizer.transform(test_df['review'])

# Convert the binary matrices into DataFrames with column names corresponding to the words
df_train_features = pd.DataFrame(X_train.toarray(), columns=vocab)
df_train_features['sentiment'] = train_df['sentiment'].values

df_test_features = pd.DataFrame(X_test.toarray(), columns=vocab)
df_test_features['sentiment'] = test_df['sentiment'].values

# Save the processed training and test data to CSV files.
# Each CSV will have 5000 binary feature columns and one sentiment column.
df_train_features.to_csv("train_data.csv", index=False)
df_test_features.to_csv("test_data.csv", index=False)
"""

def plot_learning_curve(training_data, clf, title,
                        train_sizes=[0.02, 0.1, 0.4, 1],
                        scoring="recall", cv=10):
    x_train = training_data.drop(columns=['sentiment']).values
    y_train = training_data["sentiment"].values
    train_sizes_arr, train_scores, test_scores = learning_curve(
        clf, x_train, y_train, train_sizes=train_sizes, scoring=scoring, cv=cv
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_arr, np.mean(train_scores, axis=1), 'b-', alpha=0.5, label="Training Scores")
    plt.plot(train_sizes_arr, np.mean(test_scores, axis=1), 'r-', alpha=0.5, label="Validation Scores")
    plt.xlabel("Training Examples")
    plt.ylabel(f"{scoring.capitalize()} Score")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_custom_performance(training_data, test_data, model_builder, predictor, model_name,
                            train_sizes=[500, 2500, 10000, 24999]):
    f_scores = []
    recall_scores = []
    precision_scores = []
    
    for size in train_sizes:
        training_data_subset = training_data.sample(n=size, replace=True)
        (test_f, test_r, test_p, train_f, train_r, train_p) = execute_model(
            model_builder=model_builder,
            predictor=predictor,
            training_data=training_data_subset,
            test_data=test_data,
            model_name=model_name
        )
        # In this example we plot the training metrics versus training size.
        f_scores.append(train_f)
        recall_scores.append(train_r)
        precision_scores.append(train_p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, f_scores, marker='o', linestyle='-', label='F1 Score (Train)')
    plt.plot(train_sizes, recall_scores, marker='o', linestyle='--', label='Recall (Train)')
    plt.plot(train_sizes, precision_scores, marker='o', linestyle='-.', label='Precision (Train)')
    plt.xlabel('Training Data Size')
    plt.ylabel('Score')
    plt.title(f'{model_name} Performance Metrics vs Training Size (Training Data) for class 1')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    training_data = pd.read_csv('train_data.csv', nrows=25000)
    test_data = pd.read_csv('test_data.csv', nrows=25000)

    # Run BernoulliNB training and scoring for the sklearn implementation.
    run_sklearn_bernoulli(training_data, test_data)
    
    # Plot learning curve for BernoulliNB classifier.
    plot_learning_curve(
        training_data,
        clf=BernoulliNB(),
        title="BernoulliNB Learning Curve"
    )

    # Print performance metrics for custom Naive Bayes classifier and plot performance vs training sizes.
    plot_custom_performance(
        training_data, test_data,
        model_builder=calculateProbabilities,
        predictor=classify_naive_bayes,
        model_name="Custom Naive Bayes"
    )



    # Run RandomForest training and scoring for the sklearn implementation.
    # run_sklearn_randomforest(training_data, test_data)

    # Plot learning curve for sklearn Random Forest classifier.
    # plot_learning_curve(
    #     training_data,
    #     clf=RandomForestClassifier(n_estimators=11, max_features=60, criterion="entropy"),
    #     title="Random Forest Learning Curve"
    # )

    # Print performance metrics for custom Random Forest classifier and plot performance vs training sizes.
    # plot_custom_performance(
    #     training_data, test_data,
    #     model_builder=lambda train: create_random_forest(train, 23),
    #     predictor=majority_vote_predict,
    #     model_name="Custom Random Forest"
    # )
   


    # Run sklearn adaboost training and scoring for the sklearn implementation.
    # run_sklearn_adaboost(training_data, test_data)

    # Plot learning curve for sklearn adaboost classifier.
    # plot_learning_curve(
    #     training_data,
    #     clf=AdaBoostClassifier(n_estimators = 100),
    #     title="Adaboost Learning Curve"
    # )

    # Print performance metrics for custom Adaboost.
    # execute_model(
    #     model_builder=partial(build_adaboost, M=2000),
    #     predictor=classify_adaboost,
    #     training_data=training_data,
    #     test_data=test_data,
    #     model_name="Custom AdaBoost"
    # )

    # Plot custom Adaboost performance vs training sizes:
    # plot_custom_performance(
    #     training_data,
    #     test_data,
    #     model_builder=partial(build_adaboost, M=2000),
    #     predictor=classify_adaboost,
    #     model_name="Custom AdaBoost"
    # )
