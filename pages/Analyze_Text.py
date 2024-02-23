import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud.wordcloud import WordCloud
from configs.db_configs import add_one_item
from configs.html_features import set_image, HTML_WRAPPER
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from spacy import displacy
import spacy
nlp = spacy.load('en_core_web_sm')
from collections import Counter
import neattext as nt
import neattext.functions as nfx
from textblob import TextBlob
import nltk



def get_tokens_analysis(text):
    doc_obj = nlp(text)
    tokens_stats = [(token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stop) for token in doc_obj]
    tokens_stats_df = pd.DataFrame(tokens_stats, columns=['Token', 'Shape', 'Part-of-Speech', 'Part-of-Speech Tag', 'Root', 'IsAlpha', 'IsStop'])
    return tokens_stats_df


def get_entities_tokens(text):
    doc_obj = nlp(text)
    options = {'colors' : {'MONEY' : '#3480f3'}}
    html = displacy.render(doc_obj, style='ent', options=options)
    html = html.replace('\n\n', '\n')
    entities_tokens_html = HTML_WRAPPER.format(html)
    return entities_tokens_html


def get_word_stats(text):
    text_frame_obj = nt.TextFrame(text)
    word_stats = text_frame_obj.word_stats()
    word_length_freq = text_frame_obj.word_length_freq()
    word_length_df = pd.DataFrame(word_length_freq.items(), columns=['word length', 'frequency'])
    word_length_df['word length'] = word_length_df['word length'].astype(str)
    word_length_df['word length'] = 'length ' + word_length_df['word length']
    custom_color = px.colors.sequential.Blues_r
    figure = px.pie(word_length_df, names='word length', values='frequency', title='Word Percentage Frequency by length', width=400, height=400, color_discrete_sequence=custom_color)
    return word_stats, figure


def plot_top_keywords_frequencies(text, n_top_keywords):
    preprocessed_text = nfx.remove_stopwords(text)
    try:
        blob = TextBlob(preprocessed_text)
        words = blob.words
    except:
        # These corpora are commonly used by TextBlob for various natural language processing tasks.
        nltk.download('brown')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('conll2000')
        nltk.download('movie_reviews')

        blob = TextBlob(preprocessed_text)
        words = blob.words
    finally:
        top_keywords = Counter(words).most_common(n_top_keywords)
        top_keywords_df = pd.DataFrame(top_keywords, columns=['words', 'frequency'])
        figure = px.bar(top_keywords_df, x='words', y='frequency', color='frequency', title=f'the frequency of {n_top_keywords} top keywords', width=400, height=400, color_continuous_scale='Blues')
        return figure


def get_sentence_stats(text):
    blob = TextBlob(text)
    sentences = [str(sentence) for sentence in blob.sentences]
    noun_phrases = list(blob.noun_phrases)
    sentence_stats = {
        'Number of Sentences' : len(sentences),
        'Number of Noun Phrases' : len(noun_phrases)
    }
    sentence_stats_df = pd.DataFrame(sentence_stats, index=[0])
    return sentences, noun_phrases, sentence_stats_df


def plot_tokens_pos(tokens_stats_df):
    pos_df = tokens_stats_df['Part-of-Speech'].value_counts().to_frame().reset_index()
    pos_df.columns = ['Part-of-Speech', 'Frequency']
    figure = px.bar(pos_df, x='Part-of-Speech', y='Frequency', color='Frequency', title=f'The Frequency of Tokens Part of speech', width=400, height=400, color_continuous_scale='Blues')
    return figure


def get_sentiment_analysis_res(text):
    tokenizer = AutoTokenizer.from_pretrained('stevhliu/my_awesome_model')
    inputs = tokenizer(text, return_tensors='pt')
    model = AutoModelForSequenceClassification.from_pretrained('stevhliu/my_awesome_model')
    with torch.no_grad():
        logits = model(**inputs).logits
        
    predicted_class_id = logits.argmax().item()
    model.config.id2label = {0:'Negative', 1:'Positive'}
    label = model.config.id2label[predicted_class_id]
    score = float(softmax(logits, dim=1)[0][predicted_class_id])
    sentiment_df = pd.DataFrame([[label, score]], columns=['Text Polarity', 'Belonging Probability'])
    return sentiment_df


def plot_word_frequency(text):
    wc = WordCloud(width=600, height=500).generate(text)
    fig = plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return fig


def main():
    st.title('Text Analyzer')
    im1, im2, im3 = st.columns([1, 5.3, 1])
    with im1:
        pass
    with im2:
        url = "https://i.postimg.cc/jdF1hPng/combined.png"
        html(set_image(url), height=400, width=400)
    with im3:
        pass

    text = st.text_area('Text Analyzer', placeholder='Enter your input text here ...', height=200, label_visibility='hidden')
    n_top_keywords = st.sidebar.slider('n Top keywords', 5, 15, 5, 1)

    if st.button('Analyze it'):
        if text != '':
            with st.expander('Original Text'):
                st.write(text)
                add_one_item(text, 'Text Analyzer')
            
            with st.expander('Text Analysis'):
                tokens_stats_df = get_tokens_analysis(text)
                st.dataframe(tokens_stats_df)

            with st.expander('Text Entities'):
                entities_tokens_html = get_entities_tokens(text)
                html(entities_tokens_html, height=300, scrolling=True)

            col11, col12 = st.columns(2)
            with col11:
                with st.expander('Word Statistics'):
                    word_stats_json, figure = get_word_stats(text)
                    st.json(word_stats_json)
                    st.plotly_chart(figure)
            
            with col12:
                with st.expander(f'The Frequency of {n_top_keywords} Top Keywords'):
                    figure = plot_top_keywords_frequencies(text, n_top_keywords)
                    st.plotly_chart(figure)
            
            col21, col22 = st.columns(2)
            with col21:
                with st.expander('Sentence Statistics'):
                    sentences, noun_phrases, sentence_stats_df = get_sentence_stats(text)
                    st.dataframe(sentence_stats_df)
                    st.write('Sentences:\n', sentences)
                    st.write('Noun Phrases:\n', noun_phrases)

            with col22:
                with st.expander('The Distribution of different Parts of Speech'):
                    figure = plot_tokens_pos(tokens_stats_df)
                    st.plotly_chart(figure)

            col31, col32 = st.columns(2)
            with col31:
                with st.expander('Sentiment Analysis'):
                    sentiment_df = get_sentiment_analysis_res(text)
                    st.dataframe(sentiment_df)

            with col32:
                with st.expander('Word Frequency'):
                    fig = plot_word_frequency(text)
                    st.pyplot(fig)

        else:
            st.error('Please enter a non-empty text.')
        

if __name__ == '__main__':
    main()

        