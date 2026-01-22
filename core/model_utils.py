from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def prepare_pipeline_data(df, test_size=0.3):
    train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    return train_df, val_df, test_df

def train_classifier(train_df):

    X = train_df.drop(['label', 'file_path'], axis = 1)
    y = train_df['label']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model