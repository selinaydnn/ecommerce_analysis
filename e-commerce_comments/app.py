import streamlit as st
import joblib


model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


st.title("📊 E-Ticaret Yorumları Sentiment Analizi")
user_input = st.text_area("💬 Bir ürün yorumu girin:", key="yorum_girisi")


if st.button("Analiz Et"):
    if user_input:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        sentiment = "Olumlu 😃" if prediction == 1 else "Olumsuz 😞"
        st.subheader("📝 Duygu Analizi Sonucu:")
        st.write(f"Bu yorum: **{sentiment}** olarak tahmin edildi.")
    else:
        st.warning("Lütfen bir yorum girin!")
