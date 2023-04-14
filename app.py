# Import flask libraries
from flask import Flask, render_template, request
import pickle

# Loading the two(2) pickle files: Bayes Multinomial and CountVectorizer
classifier = pickle.load(open('spam_clfer.pkl', 'rb'))
vectorizer = pickle.load(open('msg_transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		msg = [message]
		vectorized_msg = vectorizer.transform(msg).toarray()
		prediction= classifier.predict(vectorized_msg)
		return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
	app.run(debug=True)