import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from two .csv file and then merge them by "id" column
    
    Args:
    messages_filepath: str. the file path of id and messages .csv file
    categories_filepath: str. the file path of id and categories .csv file
    
    Returns:
    df (DataFrame): A dataframe of messages and categories
    """
    #reading messages and categories from .csv and merging them by 'id'
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge to df by 'id' column and return
    return pd.merge(messages, categories, on="id")

def clean_data(df):
    """
    Load dataframe and split "categories" column to seperate category columns
    
    Args:
    df: dataframe. the raw dataframe
    
    Returns:
    df (DataFrame): A dataframe of splitted categories
    """
    # taking category names from first row of dataframe
    categories = df["categories"].str.split(";", expand = True)
    row = categories.loc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: 0 if int(x) == 0 else 1)
        
        # drop 'related' column values other than '1, 0' (only different value is '2')
        categories = categories[categories.related != 2]
     
    #drop raw categories column
    df.drop(['categories'], axis=1, inplace=True)
    
    #concatenate created catories dataframe with original dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    df=df.drop_duplicates(subset='id')
    
    return df


def save_data(df, database_filename):
    """save data to sql database"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql("MessagesCategories", engine, index=False)  


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
        print(len(df))
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()