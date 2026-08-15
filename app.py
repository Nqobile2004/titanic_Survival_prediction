import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details and see if they would have survived!")

# Load and train model
@st.cache_data
def train_model():
    URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(URL)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    X = df.drop(columns=["Survived", "Cabin", "Ticket", "Name", "PassengerId"])
    y = df["Survived"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_sex, le_embarked

model, le_sex, le_embarked = train_model()

# User inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0, 500, 32)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival"):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    input_data = pd.DataFrame([[
        pclass,
        le_sex.transform([sex])[0],
        age,
        sibsp,
        parch,
        fare,
        le_embarked.transform([embarked])[0],
        family_size,
        is_alone
    ]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ SURVIVED! Probability: {proba:.2%}")
    else:
        st.error(f"❌ DID NOT SURVIVE. Probability: {1-proba:.2%}")
