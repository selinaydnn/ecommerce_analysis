import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/selinaydin/PycharmProjects/e-commerce_comments/source_doc/e-ticaret_urun_yorumlari.csv", delimiter=";")

vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(df["Metin"])


y = df["Durum"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_model = LogisticRegression()
log_model.fit(X_train, y_train)


joblib.dump(log_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model ve vektörizer başarıyla kaydedildi!")
