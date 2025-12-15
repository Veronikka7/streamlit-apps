import streamlit as st
import pandas as pd
import joblib
import os


# Путь к файлу и директория
file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


# Загружаем модель и препроцессоры
model = joblib.load(os.path.join(directory, "../data/best_model.pkl"))
scaler = joblib.load(os.path.join(directory, "../data/scaler.pkl"))
encoder = joblib.load(os.path.join(directory, "../data/encoder.pkl"))


# Заголовок приложения
st.title("Will the average person's earnings exceed the $50k threshold?")
st.write("Input the values of the features:")


# Ввод числовых признаков
numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
input_numeric = {}
for feature in numeric_features:
    input_numeric[feature] = st.number_input(f"{feature}", value=0.0)


# Ввод категориальных признаков
categorical_options = {
    'workclass': ['Local-gov', 'Private', 'Federal-gov', 'Self-emp-inc', 'State-gov', 'Self-emp-not-inc', 'Without-pay'],
    'education': ['HS-grad', 'Some-college', 'Bachelors', 'Assoc-acdm', '12th', 'Doctorate',
                  '1st-4th', '7th-8th', 'Masters', '5th-6th', '11th', '9th', 'Assoc-voc',
                  'Prof-school', '10th', 'Preschool'],
    'marital-status': ['Never-married', 'Divorced', 'Married-civ-spouse', 'Separated',
                       'Married-spouse-absent', 'Married-AF-spouse', 'Widowed'],
    'occupation': ['Farming-fishing', 'Sales', 'Prof-specialty', 'Tech-support', 'Adm-clerical',
                   'Other-service', 'Machine-op-inspct', 'Protective-serv', 'Transport-moving',
                   'Exec-managerial', 'Craft-repair', 'Handlers-cleaners', 'Priv-house-serv',
                   'Armed-Forces'],
    'relationship': ['Not-in-family', 'Own-child', 'Husband', 'Unmarried', 'Wife', 'Other-relative'],
    'race': ['Black', 'White', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Other'],
    'sex': ['Male', 'Female']
}


input_categorical = {}
for feature, options in categorical_options.items():
    input_categorical[feature] = st.selectbox(f"{feature}", options)


# Формируем DataFrame
df_numeric_input = pd.DataFrame([input_numeric])
df_categorical_input = pd.DataFrame([input_categorical])


# Масштабируем числовые признаки
df_numeric_scaled = pd.DataFrame(scaler.transform(df_numeric_input), columns=numeric_features)


# Кодируем категориальные признаки
df_categorical_encoded = pd.DataFrame(
    encoder.transform(df_categorical_input),
    columns=encoder.get_feature_names_out()
)


# Объединяем признаки
X_input = pd.concat([df_numeric_scaled, df_categorical_encoded], axis=1)


# Предсказание
if st.button("Predict"):
    prediction = model.predict(X_input)
    prediction_proba = model.predict_proba(X_input)[:, 1]
    
    st.subheader("Результат предсказания")
    st.write(f"Класс (0/1): {prediction[0]}")
    if prediction[0] == 1:
        st.write("Средний заработок превысит $50k")
    else:
        st.write("Средний заработок не превысит $50k")
    st.write(f"Вероятность положительного класса: {prediction_proba[0]:.2f}")
