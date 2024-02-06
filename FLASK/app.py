from flask import Flask, render_template, request, redirect, send_file
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
        
        return render_template('index.html', tables=[df.to_html(classes='data')],
                        img_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
