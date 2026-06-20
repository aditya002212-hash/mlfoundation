import streamlit as st
import pickle
import pandas as pd

teams=[        'Sunrisers Hyderabad',             'Kings XI Punjab',
            'Rajasthan Royals',       'Kolkata Knight Riders',
              'Mumbai Indians',         'Chennai Super Kings',
            'Delhi Daredevils', 'Royal Challengers Bangalore',
             'Deccan Chargers',              'Delhi Capitals']

cities=[     'Hyderabad',         'Mumbai',         'Jaipur',        'Chennai',
          'Delhi',      'Ahmedabad',      'Bangalore',     'Chandigarh',
   'Johannesburg',        'Kolkata',  'Visakhapatnam',      'Bengaluru',
         'Durban',           'Pune', 'Port Elizabeth',     'Dharamsala',
      'Cape Town',         'Mohali',      'Centurion',      'Abu Dhabi',
    'East London',         'Raipur',        'Cuttack',   'Bloemfontein',
         'Nagpur',         'Ranchi',         'Indore',        'Sharjah',
      'Kimberley']

model=pickle.load(open('model_lr.pkl','rb'))

st.title('IPL WIN PREDICTOR')

col1 , col2 = st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',teams)

with col2:
    bowling_team=st.selectbox('Select the bowling team',teams)

city=st.selectbox('select match city',cities)

target= st.number_input('Target')

col3 ,col4 ,col5=st.columns(3)

with col3:
    score= st.number_input('Score')

with col4:
    overs=st.number_input('over completed')

with col5:
    wickets=st.number_input('wickets')

if st.button('Predict Probablity'):
    runs_left= target-score
    ball_left=120-(overs*6)
    wicket_left=10-wickets
    crr=score/overs
    rrr=(runs_left*6)/ball_left

    input_df=pd.DataFrame({'batting_team':[batting_team],
                  'bowling_team':[bowling_team],
                  'city':[city],'current_score':[score],
                  'run_left':[runs_left],'ball_left':[ball_left],
                  'wickets_left':[wicket_left],
                  'crr':[crr],'rrr':[rrr]
                  })
    st.table(input_df)
    result=model.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team+'-- '+str(round(win*100))+'%')
    st.header(bowling_team+'-- '+str(round(loss*100))+'%')


