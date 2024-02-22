# Multi-Purpose Text Application
[![GitHub License](https://img.shields.io/github/license/amirho3einsedaghati/multi-purpose-text-application?color=yellow)](https://github.com/amirho3einsedaghati/multi-purpose-text-application/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/amirho3einsedaghati/multi-purpose-text-application?color=red)](https://github.com/amirho3einsedaghati/multi-purpose-text-application/)
[![GitHub forks](https://img.shields.io/github/forks/amirho3einsedaghati/multi-purpose-text-application?color=yellow)](https://github.com/amirho3einsedaghati/multi-purpose-text-application/forks)
[![GitHub User's stars](https://img.shields.io/github/stars/amirho3einsedaghati/multi-purpose-text-application?color=red)](https://github.com/amirho3einsedaghati/multi-purpose-text-application/stargazers)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/amirho3einsedaghati/multi-purpose-text-application?color=yellow)](https://github.com/amirho3einsedaghati/multi-purpose-text-application/pulls)
[![GitHub issues](https://img.shields.io/github/issues-raw/amirho3einsedaghati/multi-purpose-text-application?color=red)](https://github.com/amirho3einsedaghati/multi-purpose-text-application/issues)

<p>
The application consists of several components that are designed to work with text. It mentions specific functionalities such as analyzing, summarizing, and translating raw text. It can also be handy in finding keywords that provide insights into the topic being discussed.
</p>

# Components

- The Text Analyzer involves a fine-tuned version of distilbert-base-uncased to determine text polarity and the probability of belonging to a class, and uses other packages to extract token, entity, word, and sentence statistics and perform other tasks.

- The Text Summarizer utilizes a fine-tuned version of t5-small on the billsum dataset to build an abstractive summarizer that can generate new text that captures the most relevant information. To understand how well it captures all the important information, I used the ROUGE metric.

- The Text Translator employs a fine-tuned version of t5-small trained on the English-French subset of the OPUS Books dataset to convert a sequence of text from English to French. To assess the quality of the translation, I utilized the SacreBLEU metric.

- The Topic Modeling leverages CountVectorizer to vectorize the input text and utilizes singular value decomposition to decompose the bag-of-words matrix.

# Usage
1. To use <a href="https://huggingface.co/spaces/amirhoseinsedaghati/multi-purpose-text-application">the online version on HuggingFace</a>, just follow services on the sidebar.
<br></br>
2. To use it on your local machine, clone the project from <a href="https://github.com/amirho3einsedaghati/multi-purpose-text-application">this URL GitHub</a> using <pre>git clone git@github.com:amirho3einsedaghati/multi-purpose-text-application.git</pre> then navigate to the multi-purpose-text-application directory containing the Home.py file and run the Streamlit application using the command <pre>streamlit run Home.py</pre>

# License
<a href="https://github.com/amirho3einsedaghati/multi-purpose-text-application/blob/main/LICENSE">Apache-2.0</a>

# Appendix:
<a href="https://huggingface.co/spaces/amirhoseinsedaghati/multi-purpose-text-application">https://huggingface.co/spaces/amirhoseinsedaghati/multi-purpose-text-application</a>

# Maintainer Contact:
<a href="https://linktr.ee/amirhoseinsedaghati">https://linktr.ee/amirhoseinsedaghati</a>
