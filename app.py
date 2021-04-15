#import libraries
import numpy as np
from flask import Flask, render_template,request,Response
import pickle#Initialize the flask App
from utils import clean_tweets,scraper,predict
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open(r'model.pkl', 'rb'))
tv=pickle.load(open(r'tv.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/plot',methods=['POST'])
def plot_png():
    form = request.form
    print("The form is loaded")
    if request.method == 'POST':
        username = request.form['username']
        startdate= request.form['startdate']
        enddate = request.form['enddate']

    data = scraper(username, startdate, enddate)
    data = predict(data, tv, model)
    fig = create_figure(data)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    #return render_template('plot.html', name='new_plot', url='/static/images/plot.png')


def create_figure(data):
    dates = data['date'].unique()
    count_depressive = dict.fromkeys(dates, 0)
    count_non_depressive = dict.fromkeys(dates, 0)

    for i in range(len(data)):
        if data.loc[i, 'prediction'] == 1:
            count_depressive[data.loc[i, 'date']] += 1
        else:
            count_non_depressive[data.loc[i, 'date']] += 1

    # Numbers of pairs of bars you want
    N = 2
    # Data on X-axis
    # Specify the values of blue bars (height)
    blue_bar = list(count_non_depressive.values())
    # Specify the values of orange bars (height)
    orange_bar = list(count_depressive.values())
    max1 = max(blue_bar)
    max2 = max(orange_bar)
    max_overall = max(max1, max2)
    # Position of bars on x-axis
    ind = np.arange(len(blue_bar))

    fig, ax = plt.subplots(figsize=(10, 5))
    # Figure size

    # Width of a bar
    width = 0.3

    # Plotting
    ax.bar(ind - width / 2, blue_bar, width, label='Non Depressive Tweets', zorder=100)
    ax.bar(ind + width / 2, orange_bar, width, label='Depressive Tweets', zorder=100)

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('Tweet Nature over time')

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    ax.set_xticks(ind + width)
    ax.set_xticklabels((list(count_non_depressive.keys())))
    ax.set_yticks(range(0, max_overall + 1))
    # Finding the best position for legends and putting it
    ax.legend(loc='best')
    ax.grid(zorder=2)
    fig.savefig('static/images/plot.png')
    return fig


# #To use the predict button in our web-app
# @app.route('/predict',methods=['POST'])
# def predict():
#     #For rendering results on HTML GUI
#     form = request.form
#     if request.method == 'POST':
#         username = request.form['username']
#         period= request.form['period']
#         df=scraper()
#         test_array = tv.transform(clean_tweets(text)).toarray()
#         prediction = model.predict(test_array)
#         if prediction:
#             return render_template('index.html', prediction_text='The tweet is depressive, Plz reachout a Mental Health Expert')
#         else:
#             return render_template('index.html',
#                                    prediction_text='The tweet is not depressive')


if __name__ == "__main__":
    app.run(debug=True)