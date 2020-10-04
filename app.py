from flask import Flask, request, render_template
from model import predict_class

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('spamDetection.html', size=0)


@app.route('/', methods=['POST'])
def home_():
    title = request.form.get('sms')
    out = predict_class(str(title))
    print(out, '------------')
    if out:
        result = 'SPAM'
    else:
        result = 'Not-SPAM'
    print(result, len(result))
    return render_template('spamDetection.html', size=len(out), spam=result)


if __name__ == '__main__':
    app.run(port='5000')
