import pickle
import streamlit as st

model = pickle.load(open('stock_market.sav', 'rb'))

st.title('Stock Market Prediction App')
Date = st.number_input('Input Date)')
Open = st.number_input('Input Harga')
High = st.number_input('Input Harga Tertinggi')
Low = st.number_input('Input Harga Terendah')
Last = st.number_input('Input Harga Terakhir)')
Close = st.number_input('Input Harga Penutupan')

predict = ''

if st.button('Predict'):
    predict = model.predict(
        [[Date, Open, High, Low, Last, Close]]
        )
    st.write ('Turnover : ', predict)
