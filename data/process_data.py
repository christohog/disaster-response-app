import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Save messages and categories datasets into two dataFrame
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories,on='id')

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)

    # rename columns of the categories dataFrame
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[0:-2])
    
    # convert category values to 0 or 1 (from request-0 or request-1 for e.g.)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Replace categories column of df with newly created categories dataFrame
    df = df.drop(labels='categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    
    # Remove duplicated rows
    df = df.drop_duplicates(keep='first')
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_table', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()