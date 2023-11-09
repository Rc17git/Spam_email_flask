# app.py
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'  # Change this to a secure key

class EmailForm(FlaskForm):
    email_content = TextAreaField('Email Content')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = EmailForm()

    if form.validate_on_submit():
        email_content = form.email_content.data
        max_sequence_length=1000
        label=['Not Spam', 'Spam']
        with open('tokenizer_email.pickle', 'rb') as handle:
             loaded_tokenizer = pickle.load(handle)
        loaded_model=keras.models.load_model('"EmailClassifier.h5"')
        seq = loaded_tokenizer.texts_to_sequences(email_content)
        ans = pad_sequences(seq, maxlen=max_sequence_length)
        preds = loaded_model.predict(ans)
        for i in preds:
            result=label[np.around(i, decimals=0).argmax()]
        # For now, let's just print the result
        print("Email Content:", email_content)

        return render_template('result.html', result=result)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
