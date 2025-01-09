import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import plotly.express as px
import requests


# Заголовок приложения
st.title('Прогнозирование диабета')

# Загрузка данных из локального файла
file_path = r"https://raw.githubusercontent.com/Muhammad03jon/Muhammad-Olimov/refs/heads/master/diabetes%20(2).csv"# Замените на свой путь
df = pd.read_csv(file_path)

# Раздел для отображения данных
with st.expander('Данные'):
    st.write("Признаки (X):")
    X_raw = df.drop('Diabetes', axis=1)  # Убираем столбец 'Diabetes'
    st.dataframe(X_raw)

    st.write("Целевая переменная (y):")
    y_raw = df['Diabetes']
    st.dataframe(y_raw)

# Раздел для ввода данных пользователем
with st.sidebar:
    st.header("Введите признаки:")
    age = st.slider('Возраст', 21, 100, 30)
    gender = st.selectbox('Пол', ('Мужской', 'Женский'))
    bmi = st.slider('Индекс массы тела (BMI)', 10.0, 70.0, 25.0)
    sbp = st.slider('Систолическое артериальное давление (SBP)', 80, 200, 120)
    dbp = st.slider('Диастолическое артериальное давление (DBP)', 40, 120, 80)
    fpg = st.slider('Глюкоза натощак (FPG)', 50, 200, 100)
    chol = st.slider('Общий холестерин (Chol)', 100, 400, 200)
    tri = st.slider('Триглицериды (Tri)', 50, 400, 150)
    hdl = st.slider('Холестерин высокой плотности (HDL)', 20, 100, 50)
    ldl = st.slider('Холестерин низкой плотности (LDL)', 50, 200, 100)
    alt = st.slider('Аланинаминотрансфераза (ALT)', 10, 100, 20)
    bun = st.slider('Мочевина (BUN)', 5, 50, 20)
    ccr = st.slider('Креатининовый клиренс (CCR)', 30, 150, 60)
    ffpg = st.slider('Глюкоза в крови на пальце (FFPG)', 50, 200, 100)
    smoking = st.selectbox('Курите ли вы?', ('Нет', 'Да'))
    drinking = st.selectbox('Употребляете ли вы алкоголь?', ('Нет', 'Да'))
    family_history = st.selectbox('Есть ли в семье диабет?', ('Нет', 'Да'))

# Визуализация данных
st.subheader('Визуализация данных')
fig = px.scatter(
    df,
    x='BMI',
    y='Age',
    color='Diabetes',
    title='Индекс массы тела (BMI) и возраст по наличию диабета'
)
st.plotly_chart(fig)

fig2 = px.histogram(
    df, 
    x='FPG', 
    nbins=30, 
    title='Распределение глюкозы натощак (FPG)'
)
st.plotly_chart(fig2)

# Подготовка данных
data = {
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'SBP': sbp,
    'DBP': dbp,
    'FPG': fpg,
    'Chol': chol,
    'Tri': tri,
    'HDL': hdl,
    'LDL': ldl,
    'ALT': alt,
    'BUN': bun,
    'CCR': ccr,
    'FFPG': ffpg,
    'Smoking': smoking,
    'Drinking': drinking,
    'FamilyHistory': family_history
}

input_df = pd.DataFrame(data, index=[0])
input_diabetes = pd.concat([input_df, X_raw], axis=0)

with st.expander('Входные данные'):
    st.write('**Введенные данные**')
    st.dataframe(input_df)
    st.write('**Совмещенные данные** (входные данные + оригинальные данные)')
    st.dataframe(input_diabetes)

url = "https://raw.githubusercontent.com/Muhammad03jon/Muhammad-Olimov/main/catboost_model.pkl"

# Локальный путь для сохранения модели
local_model_path = "catboost_model.pkl"

model = joblib.load(local_model_path)

# Прогнозирование
input_row = input_df
prediction = model.predict(input_row)
prediction_proba = model.predict_proba(input_row)

# Вероятности предсказания
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Нет диабета', 'Диабет'])

# Результаты предсказания
st.subheader('Предсказания вероятностей')
st.dataframe(
    df_prediction_proba,
    column_config={
        'Нет диабета': st.column_config.ProgressColumn(
            'Нет диабета',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Диабет': st.column_config.ProgressColumn(
            'Диабет',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
    },
    hide_index=True
)

# Вывод финального предсказания
diabetes_status = ['Нет диабета', 'Диабет']
st.success(f"Предсказанный результат: **{diabetes_status[prediction][0]}**")
