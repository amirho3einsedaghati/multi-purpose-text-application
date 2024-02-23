import streamlit as st
import base64
import time

time_str = time.strftime('%Y%m%d-%H%M%S')

class FileDownloader(object):
    def __init__(self, data, file_ext, file_name='myfile') -> None:
        super(FileDownloader, self).__init__()
        self.data = data
        self.file_ext = file_ext
        self.file_name = file_name

    def download(self):
        b64 = base64.b64encode(self.data.encode()).decode()
        new_file_name = '{0}_{1}.{2}'.format(self.file_name, time_str, self.file_ext)
        href = """
        <a href="data: file/{}; base64,{}" download="{}">Download it</a>
        """.format(self.file_ext, b64, new_file_name)
        st.markdown(href, unsafe_allow_html=True)
        