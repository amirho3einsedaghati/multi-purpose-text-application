import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from configs.download_files import FileDownloader
from configs.db_configs import add_one_item
from streamlit.components.v1 import html
from configs.html_features import set_image
from sacrebleu.compat import corpus_bleu  
import pandas as pd



def translate_text_to_text(text, source_lang, target_lang):
    prefix = f'translate {source_lang} to {target_lang}: '
    text = prefix + text
    tokenizer = AutoTokenizer.from_pretrained('stevhliu/my_awesome_opus_books_model')
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained('stevhliu/my_awesome_opus_books_model')
    output_ids = model.generate(input_ids, max_new_tokens=len(input_ids[0]) * 3, do_sample=False, top_k=30, top_p=0.95)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text


def validate_translation(original_text, translated_text):
    return corpus_bleu(translated_text, [original_text])


def main():
    st.title('Text Translator')
    im1, im2, im3 = st.columns([1, 5.3, 1])
    with im1:
        pass
    with im2:
        url = "https://i.postimg.cc/jdF1hPng/combined.png"
        html(set_image(url), height=400, width=400)
    with im3:
        pass
    
    languages = ['English', 'French']
    source_lang = st.sidebar.selectbox('Source Language', languages)
    target_lang = st.sidebar.selectbox('Target Language', languages, index=1)
    text = st.text_area('Text Translator', placeholder='Enter your input text here ...', height=200, label_visibility='hidden')

    if st.button('translate it'):
        if text != '':
            if (source_lang == 'English' and target_lang == 'English') or (source_lang == 'French' and target_lang == 'French'):
                st.error('Expected different values for source and target languages, but got the same values!')

            else:
                with st.expander('Original Text'):
                    st.write(text)
                    add_one_item(text, 'Text Translator')

                with st.expander('Translated Text'):
                    translated_text = translate_text_to_text(text, source_lang, target_lang)
                    st.write(translated_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander('Download Translated Text'):
                        FileDownloader(translated_text, 'txt').download()

                with col2:
                    with st.expander('Translated Text Validation'):
                        bleu_score = validate_translation(text, translated_text)
                        df = pd.DataFrame({
                            'Brevity Penalty' : bleu_score.bp,
                            'length of original text' : bleu_score.ref_len,
                            'length of translated text' : bleu_score.sys_len,
                            'Ratio' : bleu_score.ratio
                        }, index=[1])
                        st.dataframe(df)

        else:
            st.error('Please enter a non-empty text.')
                

if __name__ == '__main__':
    main()