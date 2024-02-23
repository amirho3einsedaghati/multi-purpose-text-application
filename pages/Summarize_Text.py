import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from configs.download_files import FileDownloader
from configs.db_configs import add_one_item
from streamlit.components.v1 import html
from configs.html_features import set_image
from rouge import Rouge
import pandas as pd



def summarize_text(text):
    prefix = 'summarize: '
    text = prefix + text
    tokenizer = AutoTokenizer.from_pretrained('stevhliu/my_awesome_billsum_model')
    input_ids = tokenizer(text=text, return_tensors='pt')['input_ids']
    model = AutoModelForSeq2SeqLM.from_pretrained('stevhliu/my_awesome_billsum_model')

    if len(input_ids[0]) < 200:
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)
        summarized_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summarized_text
    
    elif len(input_ids[0]) > 200:
        output_ids = model.generate(input_ids, max_new_tokens=round(len(input_ids[0]) * 1/2), do_sample=False)
        summarized_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summarized_text
    

def validate_summarization(original_text, summarized_text):
    rouge_score = Rouge()
    return rouge_score.get_scores(summarized_text, original_text)


def main():
    st.title('Text Summarizer')
    im1, im2, im3 = st.columns([1, 5.3, 1])
    with im1:
        pass
    with im2:
        url = "https://i.postimg.cc/jdF1hPng/combined.png"
        html(set_image(url), height=400, width=400)
    with im3:
        pass
    
    text = st.text_area('Text Summarizer', placeholder='Enter your input text here ...', height=200, label_visibility='hidden')

    if st.button('Summarize it'):
        if text != "":
            with st.expander('Original Text'):
                st.write(text)
                add_one_item(text, "Text Summarizer")
            
            with st.expander('Summarized Text'):
                summarized_text = summarize_text(text)
                st.write(summarized_text)
            
            col1, col2 = st.columns(2)
            with col1:
                with st.expander('Download Summarized Text'):
                    FileDownloader(summarized_text, 'txt').download()
            
            with col2:
                with st.expander('Summarized Text Validation'):
                    scores = validate_summarization(text, summarized_text)
                    df = pd.DataFrame(scores[0]).T
                    df.columns = ['Recall', 'Precision', 'F1']
                    st.write(df)

        else:
            st.error('Please enter a non-empty text.')


if __name__ == '__main__':
    main()
    