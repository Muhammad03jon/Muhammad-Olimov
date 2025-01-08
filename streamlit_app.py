import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Прогнозирование наличия диабета", layout="wide")
st.title("Прогнозирование наличия диабета")

# Стилизация для визуальной привлекательности
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f0;
        padding: 20px;
    }
    .stTitle {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #1e3d58;
    }
    .stTextInput {
        font-size: 18px;
    }
    .css-1p4t15l {
        background-color: #d1e7dd !important;
    }
    </style>
""", unsafe_allow_html=True)

# Левый верхний угол - место ввода данных
with st.container():
    st.sidebar.header("Введите данные для прогноза")

    # Ввод данных с помощью слайдеров
    age = st.sidebar.slider("Возраст", min_value=18, max_value=100, value=30, step=1)
    bmi = st.sidebar.slider("Индекс массы тела (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    income = st.sidebar.slider("Доход", min_value=0, max_value=1000000, value=50000, step=1000)
    genhlth = st.sidebar.slider("Общее состояние здоровья", min_value=1, max_value=5, value=3)
    menthlth = st.sidebar.slider("Психическое здоровье", min_value=0, max_value=30, value=0)
    physhlth = st.sidebar.slider("Физическое здоровье", min_value=0, max_value=30, value=0)

    # Ввод бинарных признаков
    highbp = st.sidebar.selectbox("Есть ли у вас высокое давление?", ("Да", "Нет"))
    highchol = st.sidebar.selectbox("Есть ли у вас повышенный холестерин?", ("Да", "Нет"))
    cholcheck = st.sidebar.selectbox("Проверялись ли вы на холестерин?", ("Да", "Нет"))
    smoker = st.sidebar.selectbox("Курите ли вы?", ("Да", "Нет"))
    stroke = st.sidebar.selectbox("Был ли у вас инсульт?", ("Да", "Нет"))
    heartdisease = st.sidebar.selectbox("Есть ли у вас заболевания сердца?", ("Да", "Нет"))
    physactivity = st.sidebar.selectbox("Занимаетесь ли вы физической активностью?", ("Да", "Нет"))
    fruits = st.sidebar.selectbox("Употребляете ли вы фрукты ежедневно?", ("Да", "Нет"))
    veggies = st.sidebar.selectbox("Употребляете ли вы овощи ежедневно?", ("Да", "Нет"))
    hvyalcohol = st.sidebar.selectbox("Употребляете ли вы много алкоголя?", ("Да", "Нет"))
    anyhealthcare = st.sidebar.selectbox("Есть ли у вас доступ к медицинской помощи?", ("Да", "Нет"))
    nodocbcost = st.sidebar.selectbox("Есть ли у вас проблемы с доступом к врачу из-за стоимости?", ("Да", "Нет"))
    diffwalk = st.sidebar.selectbox("Есть ли проблемы с ходьбой?", ("Да", "Нет"))
    sex = st.sidebar.selectbox("Пол", ("Мужчина", "Женщина"))

# Преобразование данных в 0 и 1
data = {
    "Age": age,
    "BMI": bmi,
    "Income": income,
    "GenHlth": genhlth,
    "MentHlth": menthlth,
    "PhysHlth": physhlth,
    "HighBP": 1 if highbp == "Да" else 0,
    "HighChol": 1 if highchol == "Да" else 0,
    "CholCheck": 1 if cholcheck == "Да" else 0,
    "Smoker": 1 if smoker == "Да" else 0,
    "Stroke": 1 if stroke == "Да" else 0,
    "HeartDiseaseorAttack": 1 if heartdisease == "Да" else 0,
    "PhysActivity": 1 if physactivity == "Да" else 0,
    "Fruits": 1 if fruits == "Да" else 0,
    "Veggies": 1 if veggies == "Да" else 0,
    "HvyAlcoholConsump": 1 if hvyalcohol == "Да" else 0,
    "AnyHealthcare": 1 if anyhealthcare == "Да" else 0,
    "NoDocbcCost": 1 if nodocbcost == "Да" else 0,
    "DiffWalk": 1 if diffwalk == "Да" else 0,
    "Sex": 1 if sex == "Мужчина" else 0
}

# Преобразование в DataFrame
input_data = pd.DataFrame([data])

# Загрузка обученной модели
model = joblib.load("C:\Users\MSI Cyborg\MyJupyterNotebook\Уроки Zypl.ai\ML\Домашки\catboost_model.pkl")  # Замените на путь к вашей модели

# Нормализация данных
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Когда пользователь нажимает кнопку для предсказания
if st.sidebar.button('Предсказать'):
    # Применение модели
    prediction = model.predict(input_data_scaled)
    
    # Отображение результата
    if prediction[0] == 1:
        st.success("Предсказание: Диабет")
    else:
        st.success("Предсказание: Отсутствие диабета")
