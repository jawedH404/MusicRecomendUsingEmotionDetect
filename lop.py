import streamlit as st
import cv2
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model # type: ignore
from keras.preprocessing import image # type: ignore
from keras.preprocessing.image import img_to_array # type: ignore
import webbrowser



st.header('MUSIC RECOMMENDATION BASED ON EMOTION DETECTION', divider='rainbow' )

col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")

with col2:
    st.image(".\Imagesa\logo.png" , width=530, use_column_width=True)

with col3:
    st.write("")


model = load_model('emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
cv2.ocl.setUseOpenCL(False)



if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion=""
    
if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"



lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"] != "false":
        
        list = []
        count = 0
        list.clear()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count = count + 1

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)


                prediction = model.predict(cropped_img)
                label = emotion_dict[np.argmax(prediction)]  #change
                #print("label", label)
                
                max_index = int(np.argmax(prediction))
                list.append(emotion_dict[max_index])

                np.save("emotion.npy", np.array([label]))
                
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

            if count >= 20:
                break
            
        st.markdown(f"<h5 style='text-align: center; color: grey;'><b>You Seem To be {label}!</b></h5>", unsafe_allow_html=True)
      
        cap.release()
        cv2.destroyAllWindows()
        
st.write("---------------------------------------------------------------------------------------------------------------------")


btn = st.button("Recommend me songs :musical_score:")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"
  
  
  #Streamlit Customisation
st.markdown(""" <style>
header {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)