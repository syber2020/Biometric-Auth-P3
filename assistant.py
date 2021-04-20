import os
import dialogflow
from google.api_core.exceptions import InvalidArgument
import speech_recognition as sr
import pyttsx3
import datetime
from word2number import w2n
import mysql.connector
# import wikipedia
import webbrowser
from google.protobuf import json_format
import os
import time
import subprocess

from word2number import w2n

# import wolframalpha
import json
import requests

def voiceAssistant():

  print('Loading your personal assistant.')

engine=pyttsx3.init('sapi5')
def speak(text):
    engine.say(text)
    engine.runAndWait()
    engine.stop()





def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio=r.listen(source)

        try:
            statement=r.recognize_google(audio,language='en-US')
            print(f"user said:{statement}\n")

        except Exception as e:
            # speak("Pardon me, please say that again")
            return "None"
        return statement

def adminFunc(emp,id):
    while True:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'admin.json' # API key

        DIALOGFLOW_PROJECT_ID = 'your project id GCP'
        DIALOGFLOW_LANGUAGE_CODE = 'en'
        SESSION_ID = 'me'

        text_to_be_analyzed = "hi"

        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
        print(session)
        statement = takeCommand().lower()
        text_input = dialogflow.types.TextInput(text=statement, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.types.QueryInput(text=text_input)
        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            raise
        # print(response)
        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        print("Fulfillment text:", response.query_result.fulfillment_text)
        if (response.query_result.intent.display_name== 'report'):
            # print(json_format.MessageToDict(response.query_result.output_contexts.parameters))
            for context in response.query_result.output_contexts:
                for key, value in context.parameters.fields.items():
                    print(value)
                    if key == 'name':
                        print(value)
                        nm = str(value)
        speak(response.query_result.fulfillment_text)

        if (response.query_result.intent.display_name == 'report'and response.query_result.query_text!='none'):

            mp = mysql.connector.connect(host='localhost',user='root',password='sqltesting123')

            cur = mp.cursor()
            queryline = "SELECT * FROM mydb.userlogs_userlogs where user_id = \'" + id + "\' ;" 
            cur.execute(queryline)

            row = cur.fetchone()
            comment = row[4] 
            user = row[5]
            confi = row[3]
            sp = "In previous login. Confidence score for "+user+ " is " +str(confi) +". And "+ str(comment)+"."
            speak(sp)


                   
        if (response.query_result.intent.display_name== 'projid'):
            # print(json_format.MessageToDict(response.query_result.output_contexts.parameters))
            for context in response.query_result.output_contexts:
                for key, value in context.parameters.items():
                    print(value)
                    # if key == 'proid':
                    #     vl = value
                    #     vl  = w2n.word_to_num(str(value))
                    #     print(value)
                    # print(f"{key}: {value}")
            # Second element of 'OutputContexts' in query_result (a Multidict) contains parameter 'ordernumber'


    
def empFunc(emp,id):
       while True:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'private_key.json' # API key

        DIALOGFLOW_PROJECT_ID = 'your project id GCP '
        DIALOGFLOW_LANGUAGE_CODE = 'en'
        SESSION_ID = 'me'

        text_to_be_analyzed = "hi"

        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
        print(session)
        statement = takeCommand().lower()
        text_input = dialogflow.types.TextInput(text=statement, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.types.QueryInput(text=text_input)
        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            raise
        # print(response)
        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        print("Fulfillment text:", response.query_result.fulfillment_text)
        speak(response.query_result.fulfillment_text)
        if (response.query_result.intent.display_name== 'leave'):
            # print(json_format.MessageToDict(response.query_result.output_contexts.parameters))
            for context in response.query_result.output_contexts:
                for key, value in context.parameters.fields.items():
                    print(value)
                    # if (response.query_result.intent.display_name == 'report'):
                    #     pass
            mp = mysql.connector.connect(host='localhost',user='root',password='sqltesting123')

            cur = mp.cursor()
            queryline = "SELECT * FROM mydb.emp_resource_empresource where emp_id = \'" + id+ "\' ;" 
            cur.execute(queryline)

            row = cur.fetchone()
            casual = str(row[3])
            total_leave = str(row[4])
            medi_leave = str(row[5])
            sp = id +" your casual leaves are " +casual+". Your  total leave's are "+total_leave+" and your medical leave are "+ medi_leave +"."
            speak(sp)

        
def passcodeFunc():
    while True:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'key.json' #key

        DIALOGFLOW_PROJECT_ID = 'your project id'
        DIALOGFLOW_LANGUAGE_CODE = 'en'
        SESSION_ID = 'me'

        text_to_be_analyzed = "hi"

        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
        print(session)
        statement = takeCommand().lower()
        text_input = dialogflow.types.TextInput(text=statement, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.types.QueryInput(text=text_input)
        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            raise
        print(response)
        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        print("Fulfillment text:", response.query_result.fulfillment_text)
        if (response.query_result.intent.display_name== 'UserId'):
            # print(json_format.MessageToDict(response.query_result.output_contexts.parameters))
            for context in response.query_result.output_contexts:
                for key, value in context.parameters.items():
                    if key == 'id':
                        ids = value.lower()
                        print(value)
        if (response.query_result.intent.display_name== 'passco'):
            # print(json_format.MessageToDict(response.query_result.output_contexts.parameters))
            for context in response.query_result.output_contexts:
                for key, value in context.parameters.items():
                    if key == 'code':
                        vl = value
                        vl = int(vl)
                        print(value)
                    # print(f"{key}: {value}")
            # Second element of 'OutputContexts' in query_result (a Multidict) contains parameter 'ordernumber'

        speak(response.query_result.fulfillment_text)
        if (response.query_result.intent.display_name == 'passco'):

            import sqlite3

            conn = sqlite3.connect('pass.db')

            print ("Opened database successfully")
            ids = "\'"+ids +"\'"
            vl = "\'"+ str(vl) + "\'" 
            qr = 'SELECT * from security where useid == '+ ids+' AND passcode == '+vl +';'
            print(qr)
            cursor = conn.execute(qr)
            for row in cursor:
                userid = row[0]
                passcode = str(row[1])
                emp = row[2].lower()
                print(userid + "\n",passcode+"\n"+emp)
                state = userid + " you are an "+ emp
                speak(state)
                # st = str(row[1]) + " has confidence level of "+str(row[2])+"."\
                # +"Failed attempts for "+str(row[3])+" and "+str(row[4])+ " successful attempts."
                # print(st)
                # speak(st)
                engine.stop()
                if  (emp == "admin"):
                    
                    adminFunc(emp, userid)
                else:
                    
                    empFunc(emp, userid)
            
            print ("Operation done successfully")
            conn.close()
        


if __name__=='__main__':
    passcodeFunc()
