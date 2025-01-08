import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Заголовок приложения
st.title("Прогнозирование наличия диабета")

# Ввод данных с помощью виджетов Streamlit
age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
gender = st.selectbox("Пол", ("Мужчина", "Женщина"))
bmi = st.number_input("Индекс массы тела (BMI)", min_value=10.0, max_value=50.0, value=25.0)
sbp = st.number_input("Систолическое давление (SBP)", min_value=80, max_value=200, value=120)
dbp = st.number_input("Диастолическое давление (DBP)", min_value=40, max_value=120, value=80)
fpg = st.number_input("Уровень глюкозы натощак (FPG)", min_value=50, max_value=200, value=100)
chol = st.number_input("Общий холестерин (Chol)", min_value=100, max_value=400, value=200)
tri = st.number_input("Триглицериды (Tri)", min_value=50, max_value=300, value=150)
hdl = st.number_input("HDL холестерин", min_value=30, max_value=100, value=50)
ldl = st.number_input("LDL холестерин", min_value=50, max_value=200, value=100)
alt = st.number_input("Аланинаминотрансфераза (ALT)", min_value=5, max_value=50, value=20)
bun = st.number_input("Блочный азот мочевины (BUN)", min_value=5, max_value=30, value=15)
ccr = st.number_input("Креатининовый клиренс (CCR)", min_value=60, max_value=150, value=90)
ffpg = st.number_input("Сахар в крови после еды (FFPG)", min_value=50, max_value=200, value=100)
smoking = st.selectbox("Курите ли вы?", ("Да", "Нет"))
drinking = st.selectbox("Употребляете ли вы алкоголь?", ("Да", "Нет"))
family_history = st.selectbox("Есть ли в вашей семье диабет?", ("Да", "Нет"))

# Преобразование бинарных значений в 0 и 1
data = {
    "Age": age,
    "Gender": 1 if gender == "Мужчина" else 0,
    "BMI": bmi,
    "SBP": sbp,
    "DBP": dbp,
    "FPG": fpg,
    "Chol": chol,
    "Tri": tri,
    "HDL": hdl,
    "LDL": ldl,
    "ALT": alt,
    "BUN": bun,
    "CCR": ccr,
    "FFPG": ffpg,
    "Smoking": 1 if smoking == "Да" else 0,
    "Drinking": 1 if drinking == "Да" else 0,
    "FamilyHistory": 1 if family_history == "Да" else 0
}

# Преобразование в DataFrame
input_data = pd.DataFrame([data])

# Нормализация данных
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

st.title("Обучение модели с загруженными данными")

# Загружаем файл
uploaded_file = st.file_uploader("Загрузите файл CSV", type=["csv"])

if uploaded_file is not None:
    # Читаем загруженные данные
    df = pd.read_csv(uploaded_file)
    
    # Показываем данные
    st.write("Данные из файла:")
    st.write(df.head())
    
    # Предполагаем, что последняя колонка — это целевая переменная
    X = df.drop(columns=['Diabetes'])  # Замените 'target' на название вашей целевой переменной
    y = df['Diabetes']  # Целевая переменная

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Инициализация модели
    model = CatBoostClassifier(iterations=150, l2_leaf_reg=6, learning_rate=0.05, max_depth=6, rsm=0.3, verbose=0)


    # Обучение модели
    model.fit(X_train_scaled, y_train)

    # Прогнозы
    y_pred = model.predict(X_test_scaled)

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)

    # Выводим результаты
    st.write(f"Точность модели: {accuracy:.4f}")
