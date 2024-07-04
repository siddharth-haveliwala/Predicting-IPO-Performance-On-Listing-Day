from flask import Flask, jsonify, request
import pickle
import pandas as pd

# Creating a Flask app
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):

        data = "Hello world"
        return jsonify({'data': data})

@app.route('/predict/')
def price_predict():
    model = pickle.load(open('model.pkl', 'rb'))
    issue_price = request.args.get('Issue Price')
    lot_size = request.args.get('Lot Size')
    issue_price_in_cr = request.args.get('Issue Price (Rs Cr)')
    qib = request.args.get('QIB')
    nii = request.args.get('NII')
    rii = request.args.get('RII')
    total = request.args.get('TOTAL')

    test_df = pd.DataFrame({'Issue Price':[issue_price],
                            'Lot Size':[lot_size],
                            'Issue Price (Rs Cr)':[issue_price_in_cr],
                            'QIB':[qib],
                            'NII':[nii],
                            'RII':[rii],
                            'TOTAL':[total]})
    
    pred_result = model.predict(test_df)
    return jsonify({'The IPO will be a ' + str(pred_result)})

# Driver function
if __name__ == '__main__':

    app.run(debug = True)