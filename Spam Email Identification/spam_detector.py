import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ["label", "message"]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


print("Dataset Shape:", df.shape)
print(df.head())

# ----------------------------
# 2. Exploratory Data Analysis (EDA)
# ----------------------------
# Distribution of labels
sns.countplot(x=df['label'])
plt.title("Spam vs Ham Distribution")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.show()


# Message length analysis
df['msg_length'] = df['message'].apply(len)
sns.histplot(df[df['label']==0]['msg_length'], bins=50, color='blue', label='Ham', kde=True)
sns.histplot(df[df['label']==1]['msg_length'], bins=50, color='red', label='Spam', kde=True)
plt.legend()
plt.title("Message Length Distribution (Ham vs Spam)")
plt.show()
# WordClouds
spam_words = ' '.join(list(df[df['label']==1]['message']))
ham_words = ' '.join(list(df[df['label']==0]['message']))


spam_wc = WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(spam_words)
ham_wc = WordCloud(width=600, height=400, background_color='black', colormap='Blues').generate(ham_words)


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(spam_wc)
plt.axis('off')
plt.title("Spam WordCloud")


plt.subplot(1,2,2)
plt.imshow(ham_wc)
plt.axis('off')
plt.title("Ham WordCloud")
plt.show()

# ----------------------------
# 3. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# ----------------------------
# 4. Vectorization (TF-IDF)
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 5A. Naive Bayes Model
# ----------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)


print("Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb))
print("Recall:", recall_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
# ----------------------------
# 5B. Logistic Regression Model
# ----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_vec, y_train)
y_pred_log = log_model.predict(X_test_vec)


print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ----------------------------
# 6. Confusion Matrix (NB)
# ----------------------------
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# 7. Confusion Matrix (Logistic Regression)
# ----------------------------
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# ----------------------------
# 8. Sample Predictions
# ----------------------------
samples = [
"Congratulations! You've won a free lottery ticket. Claim now!",
"Hey, are we still meeting for lunch today?",
"Urgent! Your account has been suspended. Reset password here.",
"Don't forget to complete your Python homework."
]


for s in samples:
    vec = vectorizer.transform([s])
    pred = nb_model.predict(vec)[0]
    print(f"Message: {s}\nPrediction: {'Spam' if pred==1 else 'Ham'}\n")