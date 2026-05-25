import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Тест health check"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    print("✅ Health check passed")

def test_predict():
    """Тест предсказания"""
    test_data = {
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "loan_duration": 36,
        "employment_years": 5,
        "credit_history": "good",
        "purpose": "car",
        "savings": 10000
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk_category" in data
    assert "prediction" in data
    print(f"✅ Predict passed: probability={data['default_probability']:.3f}")

def test_invalid_data():
    """Тест с некорректными данными"""
    test_data = {
        "age": 150,
        "income": 50000,
        "loan_amount": 15000,
        "loan_duration": 36,
        "employment_years": 5,
        "credit_history": "good",
        "purpose": "car",
        "savings": 10000
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 422
    print("✅ Invalid data test passed")

if __name__ == "__main__":
    print("Запуск тестов...")
    test_health()
    test_predict()
    test_invalid_data()
    print("\n🎉 Все тесты пройдены!")