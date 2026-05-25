import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import sys

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_and_prepare_data, create_preprocessing_pipeline

def train_models():
    """
    Обучает и сравнивает несколько моделей
    """
    print("=" * 50)
    print("1. Загрузка данных")
    print("=" * 50)
    df = load_and_prepare_data()
    
    # Подготовка данных
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Предобработка
    preprocessor = create_preprocessing_pipeline()
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Разделение на train/val/test
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Модели для сравнения
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = []
    
    print("\n" + "=" * 50)
    print("2. Обучение и сравнение моделей")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\nОбучаем {name}...")
        model.fit(X_train, y_train)
        
        # Предсказания
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
        
        # Метрики
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_val, y_val_pred),
            'Precision': precision_score(y_val, y_val_pred),
            'Recall': recall_score(y_val, y_val_pred),
            'F1-Score': f1_score(y_val, y_val_pred),
            'ROC-AUC': roc_auc_score(y_val, y_val_proba)
        }
        results.append(metrics)
        
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    # Сравнение результатов
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("3. Сравнение моделей")
    print("=" * 50)
    print(results_df.to_string(index=False))
    
    # Выбор лучшей модели
    best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
    best_model = models[best_model_name]
    
    print(f"\nЛучшая модель: {best_model_name}")
    
    # Оценка на тестовых данных
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n" + "=" * 50)
    print("4. Финальная оценка на тестовых данных")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    
    # Сохранение модели и препроцессора
    os.makedirs('../artifacts', exist_ok=True)
    joblib.dump(best_model, '../artifacts/credit_risk_model.pkl')
    joblib.dump(preprocessor, '../artifacts/preprocessor.pkl')
    
    print("\n✅ Модель сохранена в artifacts/credit_risk_model.pkl")
    print("✅ Препроцессор сохранён в artifacts/preprocessor.pkl")
    
    return best_model, preprocessor, results_df

if __name__ == "__main__":
    train_models()