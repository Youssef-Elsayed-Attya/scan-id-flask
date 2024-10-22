from flask import  Flask , request , render_template,url_for,jsonify
from PIL import Image, ImageEnhance
import pytesseract
from ultralytics import YOLO
import re
import numpy as np

###################################################################################
def preprocess_image(image, alpha=1.1, beta=35):
    w, h = image.size
    new_width = 1024
    ratio = new_width / float(w)
    new_height = int(h * ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    gray_image = resized_image.convert('L')
    enhanced_image = ImageEnhance.Brightness(gray_image).enhance(alpha)
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(beta)

    return enhanced_image


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


###################################################################################

def find_and_format_id_in_text(text):
    pattern = r"JE (\d+)"
    match = re.search(pattern, text)
    if match:
        return f"JE {match.group(1)}"
    else:
        return 'ID not found'


def crop(img):
    yolo = YOLO('mus_weight t.pt')
    #img = cv2.imread(image_path)
    results = yolo(img)
    results = results[0]
    box = results.boxes
    current_coordinates = [round(x) for x in box.xyxy.flatten().tolist()]
    print(current_coordinates)
    (x1, y1, x2, y2) = current_coordinates
    img_array = np.array(img)
    cropped_img_array = img_array[y1:y2, x1:x2]
    cropped_img = Image.fromarray(cropped_img_array)

    return cropped_img

######################################################################################



app= Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html', appName="Id scanner")


@app.route('/predictApi',methods=['POST'])
def api():
    try:
        if 'fileup' not in request.files:
            return 'please try again, no image found'
        image = request.files.get('fileup')
        #image=cv2.imread(image)
        image = Image.open(image)
        print('there is image')
        cropped_image = crop(image)
        print('cropped image')
        pre_image= preprocess_image(cropped_image)
        print('done preprocess')
        result= extract_text_from_image(pre_image)
        print(result)
        result=find_and_format_id_in_text(result)
        return jsonify({'prediction': result})
    except:
        return jsonify({'error':'error try again'})



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        #image = cv2.imread(image)
        image = Image.open(image)
        print('there is image')
        cropped_image = crop(image)
        print('cropped image')
        pre_image = preprocess_image(cropped_image)
        print('done preprocess')
        result = extract_text_from_image(pre_image)
        print(result)
        prediction = find_and_format_id_in_text(result)

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html',appName="Intel Image Classification")

if __name__=='__main__':
    app.run(debug=True)

