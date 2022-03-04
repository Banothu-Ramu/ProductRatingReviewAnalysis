import streamlit as st
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import nltk
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
nltk.download('stopwords')
#Data cleaning and preprocessing
predictions = dict()
st.sidebar.header('Product Rating Analysis')
page = st.sidebar.selectbox("Choose your choice", ["Test with sample review", "Compare each model","Analyze Multiple Reviews"])
if page=="Test with sample review":
  st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
  st.markdown(
      f"""
      <style>
      .reportview-container {{
          background: #e6ecf5
      }}
      </style>
      """,
      unsafe_allow_html=True)
  st.markdown("<h1 style='text-align: center; color: Black;font-family:Times New Roman'>Product Rating Review System</h1>", unsafe_allow_html=True)
  st.markdown("<h2 style='font-family:Times New Roman'>Test with your own text....</h2>",unsafe_allow_html=True)
  text=st.text_area("Enter your review..",height=100)
  if st.button("Analysis"):
    br=TextBlob(text)
    result=br.sentiment.polarity
    if result==0:
       st.markdown(""" <div class="alert alert-warning" role="alert">
         You got a Neutral Review
       </div>""",unsafe_allow_html=True)
    elif result>0:
      st.markdown(""" <div class="alert alert-success" role="alert">
         You got a Positive Review
       </div>""",unsafe_allow_html=True)
    else:
       st.markdown(""" <div class="alert alert-danger" role="alert">
         You got a Negative Review
       </div>""",unsafe_allow_html=True)

elif page=="Compare each model":
  data=st.file_uploader("Enter the Data Set ",type='csv')
  if data is not None:
      flipkart_data = pd.read_csv(data)
      flipkart_data = flipkart_data.dropna(axis = 0)
      flipkart_data=flipkart_data[["review","rating"]]
      flipkart_data_pos=flipkart_data[flipkart_data["rating"].isin([4,5])]
      flipkart_data_neg=flipkart_data[flipkart_data["rating"].isin([1,2])]
      flipkart_data_filtered=pd.concat([flipkart_data_pos[:20000],flipkart_data_neg[:20000]])
      flipkart_data_filtered["r"]=1
      flipkart_data_filtered["r"][flipkart_data_filtered["rating"].isin([1,2])]= 0

      #Splitting Train and Test Data 

      X_train_data,x_test_data,Y_train_data,y_test_data=train_test_split(flipkart_data_filtered["review"],flipkart_data_filtered["r"],test_size=0.2)

      #Text Transformation using TFIDF

      tfidf_vector = TfidfVectorizer(stop_words="english")
      tfidf_vector.fit(X_train_data)
      X_train_data_new=tfidf_vector.transform(X_train_data)
      x_test_data_new=tfidf_vector.transform(x_test_data)
              
      def compare_models():
          st.write("\n\nPlease wait. This may take a few minutes")
          st.write("\nAnalyzing Logistic Regression")
          lr_model = LogisticRegression()
          lr_model.fit(X_train_data_new,Y_train_data)
          predictions['LogisticRegression'] = lr_model.predict(x_test_data_new)

          st.write("\nAnalyzing Multinomial NB")
          mul_model = MultinomialNB()
          mul_model.fit(X_train_data_new,Y_train_data)
          predictions["Multinomial"] = mul_model.predict(x_test_data_new)  
          
          st.write("\nAnalyzing SVM")
          svm_model = SVC()
          svm_model.fit(X_train_data_new,Y_train_data) 
          predictions['SVM']=svm_model.predict(x_test_data_new)

          st.write("\nAnalyzing k-NN")
          knn_model = KNeighborsClassifier(n_neighbors=1)
          knn_model.fit(X_train_data_new,Y_train_data)
          predictions["knn"] = knn_model.predict(x_test_data_new)
          st.write("\n\nCalculating Accuracy of each model.\n\n")
    
          #Model Accuracy Table
          print_results = {}
          for k,v in predictions.items():
              print_results[k] = accuracy_score(y_test_data,v)
          result_table=pd.DataFrame(list(print_results.items()), columns=["Model","Accuracy"])
          st.write(result_table)
        
          #Bar chart comparing accuracies of models
          plt.figure(figsize= (8,4))
          fig=sns.factorplot(x='Model', y='Accuracy', hue='Model', size=3,aspect=2,kind='bar', data=result_table)
          plt.title("Model accuracy")
           
          st.pyplot(fig)
          

          
      compare_models()
else:
  data=st.file_uploader("Enter the Data Set ",type='csv')
  if data is not None:
    df = pd.read_csv(data)
    vs=SentimentIntensityAnalyzer()
    df = df.dropna(axis = 0)
    stopword_list=nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    tokenizer = ToktokTokenizer()
    def cleantext(text):
      #to remove punctuations
      translator = str.maketrans('','',string.punctuation)
      text= text.translate(translator)
      #to remove special characters
      text=re.sub(r'[^A-Za-z0-9\s]','',text)
      # To remove stopwords
      tokens=tokenizer.tokenize(text)
      tokens=[token.strip() for token in tokens]
      filtered_tokens=[token for token in tokens if token not in stopword_list]
      text=' '.join(filtered_tokens)
      text=" ".join(text.split())
      return text.lower()
    df['review']=df['review'].apply(cleantext)
    df['compound']=df['review'].apply(lambda x: vs.polarity_scores(x)['compound'])
    df["label"]=df['compound'].apply(lambda c: 'Positive' if c>0 else ('Negative' if c<0 else 'Neutral'))
    x = df['label']
    df['counts'] = df['label'].map(df['label'].value_counts())
    y=df['counts']
    plt.figure(figsize= (8,4))
    plt.bar(x,y,width = 0.2)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plt.show())
    st.write("To know your negative reviews Click here..")
    if st.button('Negative Reviews'):
      st.dataframe(df[df["label"]=="Negative"][["review"]])
