from flask import Flask, render_template, request
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from asgiref.wsgi import WsgiToAsgi

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_content = request.form.get('email_content')
        max_sequence_length=1000
        label=['Not Spam', 'Spam']
        with open('tokenizer_email.pickle', 'rb') as handle:
             loaded_tokenizer = pickle.load(handle)
        loaded_model=keras.models.load_model("EmailClassifier.h5")
        seq = loaded_tokenizer.texts_to_sequences(email_content)
        ans = pad_sequences(seq, maxlen=max_sequence_length)
        preds = loaded_model.predict(ans)
        for i in preds:
            result=label[np.around(i, decimals=0).argmax()]
        print("Email Content:", email_content)

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    asgi_app = WsgiToAsgi(app)
    asgi_app.run(debug=True)
