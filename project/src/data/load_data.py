import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_prepare_data():
    """
    Загружает и подготавливает данные German Credit
    """
    # Создаём синтетические данные (реальный датасет загрузим позже)
    np.random.seed(42)
    n_samples = 1000
    
    # Признаки
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'loan_duration': np.random.randint(6, 72, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'credit_history': np.random.choice(['good', 'fair', 'poor'], n_samples),
        'purpose': np.random.choice(['car', 'education', 'home', 'business'], n_samples),
        'savings': np.random.randint(0, 50000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Целевая переменная (вероятность дефолта зависит от признаков)
    risk_score = (
        (df['age'] < 25) * 0.3 +
        (df['income'] < 30000) * 0.4 +
        (df['loan_amount'] / df['income'] > 0.5) * 0.3 +
        (df['employment_years'] < 2) * 0.2 +
        (df['credit_history'] == 'poor') * 0.4 +
        (df['credit_history'] == 'fair') * 0.2
    )
    
    df['default'] = (risk_score + np.random.random(n_samples) * 0.3) > 0.5
    df['default'] = df['default'].astype(int)
    
    return df

def create_preprocessing_pipeline():
    """
    Создаёт пайплайн для предобработки данных
    """
    numeric_features = ['age', 'income', 'loan_amount', 'loan_duration', 
                       'employment_years', 'savings']
    categorical_features = ['credit_history', 'purpose']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def get_feature_names(preprocessor):
    """
    Получает имена признаков после преобразования
    """
    numeric_features = ['age', 'income', 'loan_amount', 'loan_duration', 
                       'employment_years', 'savings']
    
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(
        ['credit_history', 'purpose']
    )
    
    return numeric_features + list(cat_features)

if __name__ == "__main__":
    # Тестирование
    df = load_and_prepare_data()
    print(f"Загружено {len(df)} записей")
    print(f"Распределение целевой переменной:\n{df['default'].value_counts()}")
    print(f"\nПервые 5 строк:\n{df.head()}")