import streamlit as st
import pandas as pd
import nltk
import re
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('wordnet')
nltk.download('punkt')


# Initialize session state variables
input_text = {'btxt':["custom tweet","@username"],'intxt':["@username / Handle","Enter custom tweet"]}
flag = False
myKey = 'my_key'
if myKey not in st.session_state:
    st.session_state[myKey] = False

if 'btntxt' not in st.session_state:
    st.session_state.btntxt=False

if 'rdf' not in st.session_state:
    st.session_state.rdf=""

if 'sentiment' not in st.session_state:
    st.session_state.sentiment = " "

if 'intext' not in st.session_state:
    st.session_state.intext = ""

if 'df' not in st.session_state:
    st.session_state.df = {"x":"","y":""}

if "dfflag" not in st.session_state:#dataframe flag for editing
    st.session_state.dfflag = False

if "waflag" not in st.session_state:#wrong ans flag for customise input
    st.session_state.waflag = False


vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
scraper = pickle.load(open('scraper.pkl','rb'))
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
dlmodel = pickle.load(open('dl_tmodel.pkl','rb'))

def changebtn():
    container_2.empty()
    if st.session_state.btntxt=="@username":
        st.session_state.btntxt="Custom Tweets"
        st.session_state[myKey] = True
    else:
        st.session_state.btntxt = "@username"
        st.session_state[myKey] = False


def remove_usernames(text):
    # Remove usernames (mentions) from the text
    return re.sub(r'@\w+', '', text)

def tokenize_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    return tokens

def normalize_tokens(tokens):
    # Normalize tokens to lowercase
    normalized_tokens = [token.lower() for token in tokens]
    return normalized_tokens
def clean_tokens(tokens):
    # Filter out non-alphanumeric tokens and usernames (mentions)
    cleaned_tokens = [token for token in tokens if token.isalnum() and not token.isdigit() and not token.startswith('@')]
    return cleaned_tokens

def remove_stopwords(tokens):
    # Remove stop words from tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    # Lemmatize tokens based on part of speech
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        pos_tag = nltk.pos_tag([token])[0][1][0].upper()
        pos_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}
        wordnet_pos = pos_map.get(pos_tag, wordnet.NOUN)
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

def filter_meaningful_tokens(tokens):
    # Filter tokens based on length and presence in WordNet
    meaningful_tokens = [token for token in tokens if (token is not None and len(token)) >= 3]
    return meaningful_tokens

def preprocess_text(text):
    # Step 1: Remove usernames
    text = remove_usernames(text)

    # Step 2: Tokenize text
    tokens = tokenize_text(text)
    
    # Step 3: Normalize tokens
    normalized_tokens = normalize_tokens(tokens)
    
    # Step 4: Clean tokens (including removing usernames)
    cleaned_tokens = clean_tokens(normalized_tokens)
    
    # Step 5: Remove stop words
    filtered_tokens = remove_stopwords(cleaned_tokens)
    
    # Step 6: Lemmatize tokens
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)

    # Step 7: Filter meaningful tokens
    meaningful_tokens = filter_meaningful_tokens(lemmatized_tokens)
    
    # Join tokens into preprocessed text
    preprocessed_text = ' '.join(meaningful_tokens)
    
    return preprocessed_text

def callme():
    with st.sidebar:
        st.markdown("Thank you for your help.")
        df=st.session_state.df
        df["x"]=df["x"].apply(preprocess_text)
        x=tokenizer.texts_to_sequences(df["x"])
        x=pad_sequences(x,maxlen=30)
        dlmodel.fit(x,df["y"])
        st.write("Model trained on new data successfully....!!")
        pickle.dump(dlmodel,open('dl_tmodel.pkl','wb'))
        st.session_state.waflag=False


# *****************Streamlit app********************
st.title("Sentiment Analysis of Tweets:")
# Create 2 columns
col1, col2= st.columns([4, 1])
st.session_state.waflag=False

with col2:
    # Toggle button
    st.write('')
    container_2 = st.empty()
    if not st.session_state.btntxt:
        st.session_state.btntxt = "Custom Tweets"
        st.session_state[myKey] =True
    container_2.button(st.session_state.btntxt,on_click=changebtn,key="b1")
    
        
        

with col1:
    # Input text box
    if st.session_state[myKey]:
        text = st.text_input(input_text['intxt'][0],"")
    else:
        text = st.text_area(input_text['intxt'][1],"")
    
    if st.session_state.intext!=text:
        st.session_state.intext=text
    # Toggle button

    if st.button('Predict', key="predict"):
        # Display the current input data
        st.subheader("RESPONSE:--")
        f=True
        if st.session_state.intext=="":
            st.write("PLEASE ENTER USERNAME....!!!")
            f=False
        if st.session_state[myKey]:
            uname=st.session_state.intext[1::]
            udetail=scraper.get_profile_info(uname)
            if udetail==None:
                st.write("ERROR :: User not found")
            else:
                tweets = scraper.get_tweets(uname, mode="user", number=50)
                st.write(f"USER FOUND")
                st.write("ID:",tweets["tweets"][0]["user"]["profile_id"])
                st.write("Upto 50 recent textual tweets fetched ::-")
                if tweets["tweets"]=="":
                    st.write("ERROR :: No tweets found. Try again after some time")
                else:
                    data={"Link To Tweet":[],"Date":[],"Content":[],"Sentiment":[],"Is_Incorrect":[]}
                    for tweet in tweets["tweets"]:
                        if tweet["text"]!= "":
                            data["Link To Tweet"].append(str(tweet["link"]))
                            data["Date"].append(str(tweet["date"]))
                            data["Content"].append(str(tweet["text"]))
                            text=preprocess_text(tweet["text"])
                            # trans=vectorizer.transform([text])
                            # ans=model.predict(trans)
                            text=tokenizer.texts_to_sequences([text])
                            text= pad_sequences(text,maxlen=30)
                            ans=(dlmodel.predict(text) > 0.5).astype(int)
                            if ans[0][0]:
                                data["Sentiment"].append(str(f"POSITIVE  \U0001F60A"))
                            else:
                                data["Sentiment"].append(str(f"NEGATIVE \U0001F922"))

                            data["Is_Incorrect"]=False
                    
                    df= pd.DataFrame(data)
                    st.session_state.df= df
                    st.session_state.dfflag=True
                    st.subheader("--------------------TWEET TABLE---------------------")
                    st.write("Please mark the checkbox which you think is incorrect or you can edit it by double clicking the cell.")
                    edit_df= st.data_editor(st.session_state.df)
                    st.markdown("Thank you for your help.")
                    re_df=edit_df[edit_df["Is_Incorrect"]==True]

                    
        else:
            t=st.session_state.intext
            li=t.split()
            if len(li)<=2:
                st.write(f"ERROR:: please write a sentence of length greater than 3 words...!!")
            else:
                
                # modify_t=preprocess_text(t)
                # trans=vectorizer.transform([modify_t])
                # ans=model.predict(trans)
                
                modify_t=preprocess_text(t)
                modify_t=tokenizer.texts_to_sequences([modify_t])
                text= pad_sequences(modify_t,maxlen=30)
                ans=(dlmodel.predict(text) > 0.5).astype(int)
                st.write(f"YOUR TEXT :: {st.session_state.intext}")
                if ans[0]==1:
                    st.write(f"SENTIMENT : POSITIVE  \U0001F60A") #U000263A
                    y=[0]

                else:
                    st.write(f" SENTIMENT : NEGATIVE \U0001F922")
                    y=[1]
                st.session_state.df = pd.DataFrame({"x":t,"y":y})
                st.write("if you found the prediction wrong in any sense please click the button below and help train the model..!!")
                st.session_state.waflag=True

                
    elif st.session_state.dfflag and st.session_state[myKey] and st.session_state.intext!="":
        st.subheader("--------------------TWEET TABLE---------------------")
        st.write("Please mark the checkbox which you think is incorrect or you can edit it by double clicking the cell.")
        st.write(f"Continue marking the cells with Incorrect data \U0001263A")
        edit_df= st.data_editor(st.session_state.df)
        st.markdown("Thank you for your help.")
        re_df=edit_df[edit_df["Is_Incorrect"]==True]
        if st.button("Update Model"):
            # Update the DataFrame in session state
            st.session_state.df = {"x":re_df["Content"],"y":[0 if (x=="POSITIVE  \U0001F60A") else 1 for x in re_df["Sentiment"]]}
            st.write("dataframe updated successfully")
            st.session_state.dfflag=False
            edit_df= st.data_editor(st.session_state.df)
            edit_df["x"]=edit_df["x"].apply(preprocess_text)
            x=tokenizer.texts_to_sequences(edit_df["x"])
            x=pad_sequences(x,maxlen=30)
            dlmodel.fit(x, edit_df["y"])
            st.session_state.df=False
            st.write("Model trained on new data successfully....!!")
            pickle.dump(dlmodel,open('dl_tmodel.pkl','wb'))


    if st.session_state.waflag:
        st.button("Wrong Prediction",on_click=callme)
        # if st.button("Wrong Prediction"):
        #     st.markdown("Thank you for your help.")
        #     dlmodel.fit(st.session_state.wdatax,st.session_state.wdatay)
        #     st.write("Model trained on new data successfully....!!")
        # st.session_state.waflag=False
