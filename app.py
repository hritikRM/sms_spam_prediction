import streamlit as st
import string
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))


st.title('SMS Spam Detector')

sms_input=st.text_area('ENTER YOUR MESSAGE')

transformed_text=transform_text(sms_input)
if st.button('Predict'):
   


   vector_input=tfidf.transform([transformed_text])

   result=model.predict(vector_input)[0]
   if result ==1:
       
       st.header('Spam')
   else:
       st.header('Not spam')