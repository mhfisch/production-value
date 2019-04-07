# import the nessecary pieces from Flask
from flask import Flask,render_template, request,jsonify,Response
import pickle
import pandas as pd

from pymongo import MongoClient, DESCENDING
client = MongoClient('localhost', 27017)
# Access/Initiate Database
db = client['event_fraud']
# Access/Initiate Table
table = db['predictions']

#Create the app object that will route our calls
app = Flask(__name__)
# Add a single endpoint that we can use for testing
@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/tabs', methods = ['GET'])
def tabs():
    return render_template('tabs.html')

@app.route('/elastic_tabs', methods = ['GET'])
def elastic_tabs():
    return render_template('elastic_tabs.html')

@app.route('/model_test', methods = ['GET'])
def model_test():
    return render_template('model_test.html')

@app.route('/db', methods = ['GET'])
def db():
    return render_template('db.html')

@app.route('/mpg', methods = ['GET'])
def mpg():
    return render_template('mpg.html')

# model = pickle.load(open('linreg.p','rb'))

@app.route('/inference',  methods = ['POST'])
def inference():
    req = request.get_json()
    print(req)
    c,h,w = req['cylinders'],req['horsepower'],req['weight']
    prediction = list(model.predict([[c,h,w]]))
    return jsonify({'c':c,'h': h,'w':w,'prediction':prediction[0] })

@app.route('/getdata',  methods = ['POST'])
def getdata():
    # req = request.get_json()
    # print(req)
    myfile = pd.read_json('test2.json')
    myname = myfile.venue_name[0]
    print (myname)
    return jsonify({'myname':myname})

@app.route('/pingtable', methods = ['POST'])
def pingtable():
    pingname = []
    pingthreat = []
    pingid = []
    for i in table.find().sort('_id', DESCENDING).limit(5):
        pingname.append(i['name'])
        pingthreat.append(i['threat'])
        pingid.append(i['object_id'])
    return jsonify({'pingname0':pingname[0], 'pingthreat0':pingthreat[0],
    'pingid0':pingid[0],
    'pingname1':pingname[1],
    'pingthreat1':pingthreat[1],
    'pingid1':pingid[1],
    'pingname2':pingname[2],
    'pingthreat2':pingthreat[2],
    'pingid2':pingid[2],
    'pingname3':pingname[3],
    'pingthreat3':pingthreat[3],
    'pingid3':pingid[3],
    'pingname4':pingname[4],
    'pingthreat4':pingthreat[4],
    'pingid4':pingid[4],

    })
#     c,h,w = req['cylinders'],req['horsepower'],req['weight']
#     prediction = list(model.predict([[c,h,w]]))
#     return jsonify({'c':c,'h': h,'w':w,'prediction':prediction[0] })
# #
# @app.route('/plot',  methods = ['GET'])
# def plot():
#     df = pd.read_csv('cars.csv')
#     data = list(zip(df.mpg,df.weight))
#     return jsonify(data)


#When run from command line, start the server
if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3336, debug = True)
