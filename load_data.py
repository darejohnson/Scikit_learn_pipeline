#load_data
import seaborn as sns

def load_tips_data():
    df = sns.load_dataset('tips')
    return df

if __name__ == "__main__":
    tips_data = load_tips_data()
    print(tips_data.head())
