# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from logging import PlaceHolder
import streamlit as st
import pandas as pd
import pickle

dataset = pd.read_csv('Student Reactions.csv')

st.title('Student Seaction Analysis')

try:
    total_posts = (st.text_input('Masukkan Total Post', max_chars=4))
    total_post = int(total_posts)

    helpful_posts = (st.text_input('Masukkan Helpful Post', max_chars=3)) 
    helpful_post = int(helpful_posts)

    nice_code_posts = (st.text_input('Masukkan Nice Code Post', max_chars=3))
    nice_code_post = int(nice_code_posts)

    collaborative_posts = (st.text_input('Masukkan Collaborative Posts', max_chars=3))
    collaborative_post = int(collaborative_posts)

    confused_posts = (st.text_input('Masukkan Confused Post', max_chars=3))
    confused_post = int(confused_posts)

    creative_posts = (st.text_input('Masukkan Creative Post', max_chars=3))
    creative_post = int(creative_posts)

    Bad_posts = (st.text_input('Masukkan Bad Post', max_chars=3))
    Bad_post  = int(Bad_posts)

    amazing_posts = (st.text_input('Masukkan Amazing Post', max_chars=3))
    amazing_post = int(amazing_posts)

    timeonlines = (st.text_input('Masukkan Timeonline Post', max_chars=6))
    timeonline = int(timeonlines)

    sk1_classroom = float(st.text_input('Masukkan sk1 classroom', placeholder="Masukan nilai 0 - 10", max_chars=2))

    sk2_classroom = float(st.text_input('Masukkan sk2 classroom', placeholder="Masukan nilai 0 - 10", max_chars=2))

    sk3_classroom = float(st.text_input('Masukkan sk3 classroom', placeholder="Masukan nilai 0 - 10", max_chars=2))

    sk4_classroom = float(st.text_input('Masukkan sk4 classroom', placeholder="Masukan nilai 0 - 10", max_chars=2))

    sk5_classroom = float(st.text_input('Masukkan sk5 classroom', placeholder="Masukan nilai 0 - 10", max_chars=2 ))

    avg = (sk1_classroom + sk2_classroom + sk3_classroom + sk4_classroom + sk5_classroom) / 5
    st.markdown(f'<p class="big-font">Average sk1-sk5 : {avg}</p>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        text-align: center;
        color: #270;
        background-color: #DFF2BF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if avg <= 5:
        hasil = ('Not Approved')
    else:
        hasil = ('Approved')

    st.markdown(f'<p class="big-font">Hasil Klasifikasi : {hasil}</p>', unsafe_allow_html=True)
 
    new_val = pd.DataFrame([[total_post, helpful_post, nice_code_post, collaborative_post, confused_post, creative_post, Bad_post, amazing_post, timeonline, sk1_classroom, sk2_classroom,  sk3_classroom, sk4_classroom, sk5_classroom]])

    infile = open('model_decision_tree.pkl', 'rb')
    dt_model = pickle.load(infile)
    infile.close()

    infile = open('enc.pkl', 'rb')
    encoding = pickle.load(infile)
    infile.close()

    new_val = encoding.transform(new_val)

    y_pred = dt_model.predict(new_val)


    
except:
    pass


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
