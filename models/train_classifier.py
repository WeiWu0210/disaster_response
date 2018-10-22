import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline,FeatureUnion
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import  classification_report,accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loads data from database
    Args:
        database_filepath: path to database
    Returns:
        X: feature
        y: labels
        col_names: label names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath) 
    df = pd.read_sql_table('message',engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)
    col_names = y.columns
    return X,y,col_names


def tokenize(text):
    """
    Tokenize text.
    Args:
        text: text string
    Returns:
        clean_tokens: array of clean tokens (str)
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Build MultiOutputClassifier model 
    Args:
        None
    Returns:
        cv: Best model based on grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0)))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance on test data set
    Args:
        model: Trained model
        X_test: test features
        Y_test: test labels
        category_names: label (category) names
    """
    y_pred = model.predict(X_test).astype(int)
    Y_test = Y_test.astype(int)
    print(classification_report(y_pred, Y_test, target_names=category_names))
    accu_score_all= []
    for i in range(len(category_names)):
        score_accu = accuracy_score(Y_test.iloc[:,i], y_pred[:,i])
        print('Accuracy score  for column',category_names[i], ': ',score_accu)
        accu_score_all.append(score_accu)
    print ('Average accuracy score: ',np.mean(accu_score_all))


def save_model(model, model_filepath):
    """
    Save the model to  pickle
    Args:
        model: Trained model
        model_filepath: path where to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()