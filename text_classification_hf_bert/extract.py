# use huggingface to retrieve the dataset

from datasets import load_dataset
import pandas as pd

class Extract():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def extract(self):
        # get imdb dataset from huggingface
        dataset = load_dataset("ag_news")
        
        # Convert train split to pandas dataframe
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Combine train and test data if needed
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # save dataframe to csv
        combined_df.to_csv('imdb_dataset.csv', index=False)

        # print the number of rows in the dataframe
        print(f"Number of rows in the dataframe: {len(combined_df)}")

        # print the columns in the dataframe
        print(f"Columns in the dataframe: {combined_df.columns.tolist()}")
        
        return combined_df
    


# Add main function

if __name__ == "__main__":
    extract = Extract("imdb")
    df = extract.extract()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())


