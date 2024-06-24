from flask import Flask, render_template, url_for, request, session
import sqlite3
import secrets
import pandas as pd
import pickle
from typing import Tuple
import tensorflow as tf
import numpy as np
import sqlite3
from flask import Flask, render_template, request
import csv
import os
from googletrans import Translator
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
from flask import Flask, jsonify, render_template, request
import google.generativeai as genai
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import sqlite3

# # Load prediction model
# stemmer = LancasterStemmer()
# model = load_model('chatbot_model.h5')
# intents = json.loads(open('intents2.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# labels = pickle.load(open('labels.pkl', 'rb'))


# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
#     return sentence_words


# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print("found in bag: %s" % w)
#     return np.array(bag)


# def predict_class(sentence):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
#     return return_list

# Load Gemini AI model
genai.configure(api_key='AIzaSyAg6UtggTP8rYwWQ-oBhJQf7xDa7SyyhpE')
gemini_model = genai.GenerativeModel('gemini-pro')
chat = gemini_model.start_chat(history=[])

connection = sqlite3.connect('database.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS admin (Id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS user (Id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS video (Id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT,  file TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS audio (Id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, file TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS pdf (Id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, file TEXT)"""
cursor.execute(command)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
chat_history = []

model3 = pickle.load(open("0-3.sav", "rb"))
model11 = pickle.load(open("4-11.sav", "rb"))
MODEL_PATH = 'image_model.tflite'

# def get_interpreter(model_path: str) -> Tuple:
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
    
#     return interpreter, input_details, output_details

# def predict(image_path: str) -> int:
#     interpreter, input_details, output_details = get_interpreter(MODEL_PATH)
#     input_shape = input_details[0]['shape']
#     img = tf.io.read_file(image_path)
#     img = tf.io.decode_image(img, channels=3)
#     img = tf.image.resize(img, (input_shape[2], input_shape[2]))
#     img = tf.expand_dims(img, axis=0)
#     resized_img = tf.cast(img, dtype=tf.uint8)
    
#     interpreter.set_tensor(input_details[0]['index'], resized_img)
#     interpreter.invoke()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     results = np.squeeze(output_data)
#     return np.argmax(results, axis=0)

# # Set Gradient AI credentials
# os.environ['GRADIENT_ACCESS_TOKEN'] = "7Vx1Gd1ImbDszNHEyQ7s8Cf7iITlxWR5"
# os.environ['GRADIENT_WORKSPACE_ID'] = "e43d3497-89d0-4d9e-80d6-da0429a74495_workspace"

# # Function to parse CSV data
# def parse_csv_data(csv_path):
#     rows_to_keep = []
#     with open(csv_path, encoding="utf-8-sig") as f:
#         reader = csv.DictReader(f, delimiter=",")
#         last_row = None
#         for row in reader:
#             if "Bot" == row["name"] and last_row is not None:
#                 rows_to_keep.append(last_row)
#                 rows_to_keep.append(row)
#                 last_row = None
#             else:
#                 last_row = row
#     return rows_to_keep

# # Function to format training data
# def format_training_data(rows_to_keep, role_play_prompt):
#     lines = []
#     for i in range(0, len(rows_to_keep), 0):
#         prompt = rows_to_keep[i]["line"].replace('"', '\\"')
#         response = rows_to_keep[i + 1]["line"].replace('"', '\\"')

#         template = (
#             f"<s>### Instruction:\n{role_play_prompt}\n\n###Input:\n{prompt}\n\n### Response:\n{response}</s>"
#         )

#         lines.append({"inputs": template})

#     return lines

# # Fine-tune the model adapter
# def fine_tune_adapter(adapter, training_data_chunks):
#     for i, chunk in enumerate(training_data_chunks):
#         try:
#             print(f"Fine-tuning chunk {i} of {len(training_data_chunks) - 1}")
#             adapter.fine_tune(samples=chunk)
#         except Exception as error:
#             print(f"*** Error processing chunk {i}: {error}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/echatbot')
def echatbot():
    return render_template('echatbot.html')

@app.route('/suggest')
def suggest():
    return render_template('suggest.html')

@app.route('/admin')
def admin():
    return render_template('adminlog.html')

@app.route('/health')
def health():
    return render_template('health.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/user')
def user():
    return render_template('userlog.html')

@app.route('/python', methods=['GET', 'POST'])
def python():
    if request.method == 'POST':
        score = 0
        answers = {
            'q1': 'b',
            'q2': 'c',
            'q3': 'a',
            'q4': 'd',
            'q5': 'b',
            'q6': 'c',
            'q7': 'a',
            'q8': 'b',
            'q9': 'c',
            'q10': 'd'
        }
        for question, answer in answers.items():
            user_answer = request.form.get(question)
            if user_answer == answer:
                score += 1
        
        msg =  f'Your score is {score}/10. Thank you for taking the quiz!'
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        query = "SELECT * FROM video WHERE subject = 'python'"
        cursor.execute(query)
        video = cursor.fetchall()

        query = "SELECT * FROM pdf WHERE subject = 'python'"
        cursor.execute(query)
        pdf = cursor.fetchall()

        query = "SELECT * FROM audio WHERE subject = 'python'"
        cursor.execute(query)
        audio = cursor.fetchall()

        return render_template('python.html', msg=msg, video=video, audio=audio, pdf=pdf)
    return render_template('python.html')

@app.route('/html', methods=['GET', 'POST'])
def html():
    if request.method == 'POST':
        score = 0
        answers = {
            'q1': 'a',
            'q2': 'a',
            'q3': 'a',
            'q4': 'b',
            'q5': 'b',
            'q6': 'a',
            'q7': 'b',
            'q8': 'a',
            'q9': 'b',
            'q10': 'a'
        }
        for question, answer in answers.items():
            user_answer = request.form.get(question)
            if user_answer == answer:
                score += 1
        
        msg =  f'Your score is {score}/10. Thank you for taking the quiz!'

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        query = "SELECT * FROM video WHERE subject = 'html'"
        cursor.execute(query)
        video = cursor.fetchall()

        query = "SELECT * FROM pdf WHERE subject = 'html'"
        cursor.execute(query)
        pdf = cursor.fetchall()

        query = "SELECT * FROM audio WHERE subject = 'html'"
        cursor.execute(query)
        audio = cursor.fetchall()

        return render_template('html.html', msg=msg, video=video, audio=audio, pdf=pdf)
    return render_template('html.html')

@app.route('/css', methods=['GET', 'POST'])
def css():
    if request.method == 'POST':
        score = 0
        answers = {
            'q1': 'c',
            'q2': 'a',
            'q3': 'b',
            'q4': 'd',
            'q5': 'c',
            'q6': 'b',
            'q7': 'a',
            'q8': 'c',
            'q9': 'b',
            'q10': 'a'
        }
        for question, answer in answers.items():
            user_answer = request.form.get(question)
            if user_answer == answer:
                score += 1
        
        msg =  f'Your score is {score}/10. Thank you for taking the quiz!'

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        query = "SELECT * FROM video WHERE subject = 'css'"
        cursor.execute(query)
        video = cursor.fetchall()

        query = "SELECT * FROM pdf WHERE subject = 'css'"
        cursor.execute(query)
        pdf = cursor.fetchall()

        query = "SELECT * FROM audio WHERE subject = 'css'"
        cursor.execute(query)
        audio = cursor.fetchall()

        return render_template('css.html', msg=msg, video=video, audio=audio, pdf=pdf)
    return render_template('css.html')

@app.route('/js', methods=['GET', 'POST'])
def js():
    if request.method == 'POST':
        score = 0
        answers = {
            'q1': 'b',
            'q2': 'c',
            'q3': 'a',
            'q4': 'd',
            'q5': 'b',
            'q6': 'c',
            'q7': 'a',
            'q8': 'b',
            'q9': 'c',
            'q10': 'd'
        }
        for question, answer in answers.items():
            user_answer = request.form.get(question)
            if user_answer == answer:
                score += 1
        
        msg = f'Your score is {score}/10. Thank you for taking the quiz!'

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        query = "SELECT * FROM video WHERE subject = 'js'"
        cursor.execute(query)
        video = cursor.fetchall()

        query = "SELECT * FROM pdf WHERE subject = 'js'"
        cursor.execute(query)
        pdf = cursor.fetchall()

        query = "SELECT * FROM audio WHERE subject = 'js'"
        cursor.execute(query)
        audio = cursor.fetchall()

        return render_template('js.html', msg=msg, video=video, audio=audio, pdf=pdf)
    return render_template('js.html')

@app.route('/adminlog', methods=['GET', 'POST'])
def adminlog():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        email = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM admin WHERE email = '"+email+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()

        if result:
            return render_template('adminlog.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
    return render_template('index.html')

@app.route('/adminreg', methods=['GET', 'POST'])
def adminreg():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        cursor.execute("INSERT INTO admin VALUES (NULL, '"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')


@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        email = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM user WHERE email = '"+email+"' AND password= '"+password+"'"
        cursor.execute(query)
        result = cursor.fetchall()

        if result:
            return render_template('userlog.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        cursor.execute("INSERT INTO user VALUES (NULL, '"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        subject = request.form['subject']
        link = request.form['link']

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        cursor.execute("INSERT INTO video VALUES (NULL, '"+subject+"', '"+link+"')")
        connection.commit()

        return render_template('adminlog.html')
       
    return render_template('adminlog.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        subject = request.form['subject']
        file = request.form['file']

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        cursor.execute("INSERT INTO audio VALUES (NULL, '"+subject+"', '"+file+"')")
        connection.commit()

        return render_template('audio.html')
       
    return render_template('audio.html')

@app.route('/pdf', methods=['GET', 'POST'])
def pdf():
    if request.method == 'POST':
        subject = request.form['subject']
        file = request.form['file']

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        cursor.execute("INSERT INTO pdf VALUES (NULL, '"+subject+"', '"+file+"')")
        connection.commit()

        return render_template('pdf.html')
       
    return render_template('pdf.html')

@app.route('/Three_year', methods=['GET', 'POST'])
def Three_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'middle eastern':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'White European':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'black':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'asian':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'south asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Native Indian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if df['etnicity'] == 'Others':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])	
        if df['etnicity'] == 'mixed':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if df['etnicity'] == 'Pacifica':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        data.append(int(df['work']))
        data.append(int(df['mh']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model3.predict([data])
        if Index == 0:
            prediction = 'Normal'
            return render_template('sad.html')
        else:
            prediction = 'Mental stress'
            return render_template('happy.html')
        print(prediction)

        #return render_template('index.html', name=name, email=email,  prediction=prediction)
    return render_template('index.html')

@app.route('/Eleven_year', methods=['GET', 'POST'])
def Eleven_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'Others':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Middle Eastern':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'White-European':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Black':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'South Asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])	
        if df['etnicity'] == 'Asian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Pasifika':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if df['etnicity'] == 'Turkish':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        data.append(int(df['work']))
        data.append(int(df['mh']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model11.predict([data])
        if Index == 0:
            prediction = 'Normal'
            return render_template('sad.html')
        else:
            prediction = 'Mental stress'
            return render_template('happy.html')
        print(prediction)

        #return render_template('index.html', name=name, email=email,  prediction=prediction)
    return render_template('index.html')

@app.route('/Image', methods=['GET', 'POST'])
def Image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        path = 'static/test/'+filename
        Index = predict(path)

        print(name)
        if Index == 0:
            prediction = 'Normal'
            return render_template('sad.html')
        else:
            prediction = 'Mental stress'
            return render_template('happy.html')
        print(prediction)
##
##        return render_template('index.html', name=name, email=email, prediction=prediction, img='http://127.0.0.1:5000/'+path)
    return render_template('index.html')

from googletrans import Translator

# Define separate chat_history variables for each function
chat_history1 = []
chat_history2 = []

@app.route('/analyse1', methods=['GET', 'POST'])
def analyse1():
    global chat_history1
    if request.method == 'POST':
        import csv
        user_input = request.form['query']
        # Get response from Gemini AI model
        gemini_response = chat.send_message(user_input)

        # Translate the response to English
        translator = Translator()
        english_text = translator.translate(gemini_response.text, dest='en').text
        print(english_text)

        data = english_text
        result = []
        for row in data.split('*'):
            if row.strip() != '':
                result.append(row)
        print(result)

        if not os.path.exists('records.csv'):
            f = open('records.csv', 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(['user', 'bot'])
            f.close()

        f = open('records.csv', 'a', newline='')
        writer = csv.writer(f)
        writer.writerow([user_input, str(result)])
        f.close()

        import random
        filename = 'voice/' + str(random.randint(1111, 9999)) + '.mp3'
        myobj = gTTS(text=str(result), lang='en', tld='com', slow=False)
        myobj.save(filename)
        print('\n------------Playing--------------\n')
        song = MP3(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()

        chat_history1.append([user_input, result])

        import csv
        f = open('HOSPITAL.csv', 'r')
        data = csv.reader(f)
        header = next(data)
        try:
            for row in data:
                if row[1] in user_input:
                    name = row[0]
                    Link = row[2]
                    break

            return render_template('echatbot.html', eng_text=english_text, status="Sever", chat_history=chat_history1, name=name, Link=Link)
        except:
            return render_template('echatbot.html', eng_text=english_text, chat_history=chat_history1)
    return render_template('echatbot.html')

@app.route('/analyse2', methods=['GET', 'POST'])
def analyse2():
    global chat_history2
    if request.method == 'POST':
        import csv
        user_input = request.form['query']
        # Get response from Gemini AI model
        gemini_response = chat.send_message(user_input)

        data = gemini_response.text
        result = []
        for row in data.split('*'):
            if row.strip() != '':
                result.append(row)
        print(result)

        if not os.path.exists('records.csv'):
            f = open('records.csv', 'w', newline = '')
            writer = csv.writer(f)
            writer.writerow(['user', 'bot'])
            f.close()
            
        f = open('records.csv', 'a', newline = '')
        writer = csv.writer(f)
        writer.writerow([user_input, str(result)])
        f.close()

        import random
        filename = 'voice/'+str(random.randint(1111, 9999))+'.mp3'
        myobj = gTTS(text=str(result), lang='en', tld='com', slow=False)
        myobj.save(filename)
        print('\n------------Playing--------------\n')
        song = MP3(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()

        kan_text = None
        # Update chat history with both responses
        translator = Translator()
        from_lang = 'en'
        to_lang = 'kn'
        text_to_translate = translator.translate(str(result), src=from_lang, dest=to_lang)
        kan_text = text_to_translate.text
        print(kan_text)

        myobj = gTTS(text=kan_text, lang='kn', tld='com', slow=False)
        myobj.save(filename)
        print('\n------------Playing--------------\n')
        song = MP3(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()

        chat_history2.append([user_input, result, kan_text])

        import csv
        f = open('HOSPITAL.csv', 'r')
        data = csv.reader(f)
        header = next(data)
        try:
            for row in data:
                if row[1] in user_input:
                    name = row[0]
                    Link = row[2]
                    break
            
            return render_template('chatbot.html', kan_text=kan_text, status="Sever", chat_history=chat_history2, name=name, Link=Link)
        except:
            return render_template('chatbot.html', kan_text=kan_text, chat_history=chat_history2)
    return render_template('chatbot.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)
