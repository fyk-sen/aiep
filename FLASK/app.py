from flask import Flask, render_template, request, redirect, send_file
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

import pandas as pd
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64

app = Flask(__name__)

def generate_image(df):
    image_data = df['PIXEL_COLOR'].values.reshape(69, 112)
    
    fig, ax = plt.subplots()
    ax.imshow(image_data, cmap='gray', interpolation='nearest', origin='lower')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_title(f'IMG_ID: {df.IMG_ID.iloc[0]}')

    # Save the figure to a BytesIO object
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)

    # Encode image data in base64
    encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
    return encoded_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        
        # Generate the image
        image_data = generate_image(df)

        prediction_key = ApiKeyCredentials(in_headers={"Prediction-key": '6d477f43feea4a2199b13c90b55da503'})
        ENDPOINT = 'https://aieprojecttest.cognitiveservices.azure.com/'
        project_id = 'd205061d-5fa6-429f-991e-8ade32e785d5'
        published_name = 'Test'

        predictor = CustomVisionPredictionClient(ENDPOINT, prediction_key)

        image_bytes = base64.b64decode(image_data)

        predict = predictor.classify_image_with_no_store(project_id, published_name, image_bytes)

        predictionlist = []
        for pred in predict.predictions:

            tag = str(pred.tag_name)
            tag.replace("''", '')
            predictionlist.append(tag)

            acc = '{0:.2f}%'.format(pred.probability * 100)
            predictionlist.append(acc)
        

        winner = predictionlist[0]
        winacc = predictionlist[1]
        loser = f'{predictionlist[2]} predicted at {predictionlist[3]}'

        if predictionlist[0] == 'No Lasso':
            color = 'green'
        else:
            color = 'red'


        return render_template('index.html', tables=[df.to_html(classes='data')],
                        img_data=image_data, winner = winner, winacc = winacc, loser = loser, color = color)
    

if __name__ == '__main__':
    app.run(debug=True)
