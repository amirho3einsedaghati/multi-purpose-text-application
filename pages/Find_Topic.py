import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
import regex as re
from configs.db_configs import add_one_item
from streamlit.components.v1 import html
from configs.html_features import set_image
from configs.download_files import FileDownloader



def preprocess_text(text):
    vectorizer = CountVectorizer(stop_words='english')
    vector = vectorizer.fit_transform([text]).todense()
    vocab = np.array(vectorizer.get_feature_names_out())
    U, s, Vh = linalg.svd(vector, full_matrices=False)
    return vocab, U, s, Vh


def show_topics(text, num_top_words):
    vocab, U, s, Vh = preprocess_text(text)
    pattern = '\d+'
    top_words = lambda Vh: [vocab[i] for i in np.argsort(Vh)[:-num_top_words-1:-1]]
    topic_words = top_words(Vh[0])
    topic_words = ' '.join(topic_words)
    return ' '.join([re.sub(pattern, '', word) for word in topic_words.split()])


def main():
    st.title('Topic Modeling by Top Keywords')
    im1, im2, im3 = st.columns([1, 5.3, 1])
    with im1:
        pass
    with im2:
        url = "https://i.postimg.cc/jdF1hPng/combined.png"
        html(set_image(url), height=400, width=400)
    with im3:
        pass

    text = st.text_area('Find Topic', placeholder='Enter your input text here ...', height=200, label_visibility='hidden')
    num_top_words = st.sidebar.slider('Number of Top Keywords', min_value=5, max_value=20, step=1, value=10)
    
    if st.button('Find Topic'):
        if text != '':
            with st.expander('Original Text'):
                st.write(text)
                add_one_item(text, 'Topic Modeling')

            with st.expander(f'Show Topic by {num_top_words} Top Keywords'):
                topic_words = show_topics(text, num_top_words)
                st.write(topic_words)

            with st.expander('Download Topic words'):
                FileDownloader(data=topic_words, file_ext='txt').download()


if __name__ == '__main__':
    main()