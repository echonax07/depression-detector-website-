#import libraries
import numpy as np
from flask import Flask, render_template,request,Response
import pickle#Initialize the flask App
from utils import clean_tweets,scraper,predict
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import  gridspec
app = Flask(__name__)
model = pickle.load(open(r'model.pkl', 'rb'))
tv=pickle.load(open(r'tv.pkl', 'rb'))



#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/show_history')
def show_history():
    return render_template('show_history.html')

@app.route('/classify_tweet')
def classify_tweet():
    return render_template('classify_tweet.html')




@app.route('/plot',methods=['POST'])
def plot_png():
    form = request.form
    print("The form is loaded")
    if request.method == 'POST':
        username = request.form['username']
        count= request.form['count']


    data,start_date,end_date = scraper(username, count)
    data = predict(data, tv, model)
    fig = create_figure(data,start_date,end_date)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    #return render_template('plot.html', name='new_plot', url='/static/images/plot.png')


def create_figure(data,start_date,end_date):
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    x = list(range(1, len(data) + 1))

    # plot line plot
    ax0.plot(x, data['prediction'], color='blue')
    ax0.set_yticks(range(0, 2))
    # Finding the best position for legends and putting it
    ax0.legend(loc='best')
    ax0.set_xlabel('Tweet Count')
    ax0.set_ylabel('Tweet Nature')
    ax0.set_title('Tweet Nature in the timeperiod: {} to {}'.format(start_date, end_date))

    labels = 'Depressive', 'Non Depressive'
    sizes = [100 * data['prediction'].tolist().count(1) / len(data),
             100 * data['prediction'].tolist().count(0) / len(data)]
    colors = ["Red", "Blue"]
    # plot pie chart
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Ratio')
    fig.savefig('static/images/plot.png')
    return fig


#To use the predict button in our web-app
@app.route('/classify',methods=['POST'])
def classify():
    #For rendering results on HTML GUI
    form = request.form
    if request.method == 'POST':
        text = [request.form['tweet']]
        print("classify is loaded")
        test_array = tv.transform(clean_tweets(text)).toarray()
        prediction = model.predict(test_array)
        if prediction:
            return render_template('classify_tweet.html', prediction_text='The tweet is depressive, Plz reachout a Mental Health Expert')
        else:
            return render_template('classify_tweet.html',
                                   prediction_text='The tweet is not depressive')


if __name__ == "__main__":
    app.run(debug=True)