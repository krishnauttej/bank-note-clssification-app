

from flask import Flask,render_template,url_for,request
import pickle
import numpy as np



classifier = pickle.load(open('classifier.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():

    if request.method == 'POST':
        features=[float(x) for x in request.form.values()]
        final=[np.array(features)]
        prediction=classifier.predict(final)
        return render_template('result.html',prediction_text=prediction)
    
    
    
    

    
    
if __name__ == '__main__':
	app.run(debug=True)    