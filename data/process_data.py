import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets 
    Args:
        messages_filepath: filepath to the messages dataset
        categories_filepath: filepath to the categories dataset
    Returns:
        df: Merged Pandas dataframe
    """
    # read in file
    messages =  pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataset
    Args:
        df: Merged pandas dataframe
    Returns:
        df: Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]
    #print(row)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)
    
    dummy_related = pd.get_dummies(categories['related'])
    dummy_related.columns=['related_0','related_1','related_2']
    dummy_related = dummy_related.astype(str)
    categories=categories.join(dummy_related)
    categories.drop('related',axis=1,inplace=True)
    #categories.head()
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df=df.join(categories)
    # drop duplicates
    #print(df.shape)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save cleaned dataset into a sqlite database
    Args:
        df:  cleaned dataframe
        database_filename: Name of the database file
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('message', engine, index=False,if_exists ='replace') 
    

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