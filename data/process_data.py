import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the two csv files and merge them together into a data frame

    Args:
        messages_filepath: Filepath to the messages csv file
        categories_filepath: Filepath to the categories csv file

    Returns:
        df: Dataframe with the uncleaned data
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    '''
    Cleans inputted dataframe. The single categories column
    is expanded so that each category has its own column.
    The records of the category columns are made numerical.

    Args:
        df: Dataframe with the uncleaned data

    Returns:
        df: Cleaned dataframe
    '''

    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    categories.columns = category_colnames

    for column in categories:
        categories[column] = pd.Series(categories[column]).str[-1]
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()

    # remove child_alone column because it only contains 0's
    df.drop('child_alone', axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Saves the dataframe to a database file. the name of the table
    is set to 'Messages'.  

    Args:
        df: Dataframe with the cleaned data
        database_filename: The name for the database file to be saved

    Returns:
        None
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False) 


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