import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_table',engine.connect())
    # For now, only message is kept for training 
    X = df[['message']]#,'genre']]
    Y = df.drop(labels=['id','message','original','genre'],axis=1)
    return X,Y


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    
    pipeline = Pipeline([        
        ('features',FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf_rf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf_rf__estimator__min_samples_split': [3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
    
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)        

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()