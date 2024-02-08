from flask import  Flask , request , render_template,url_for,jsonify
import pytesseract
import cv2
from ultralytics import YOLO
import re

###################################################################################
def preprocess_image(image, alpha=1.1, beta=35):
    h, w = image.shape[:2]
    new_width = 1024
    ratio = new_width / float(w)
    new_height = int(h * ratio)

    # تغيير حجم الصورة
    resized_image = cv2.resize(image, (new_width, new_height))

    # تحويل الصورة إلى رمادية وتعزيزها
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    return enhanced_image


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# معالجة الصورة المقصوصة واستخراج النص
# processed_image = preprocess_image(cropped_image)
# extracted_text = extract_text_from_image(processed_image)

# عرض الصورة المقصوصة والنص المستخرج
# show_image(processed_image, title='Processed Cropped Image')
# print("Extracted Text:\n", extracted_text)

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
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

#cropped_image = crop('photo_2023-11-10_22-20-08.jpg')

# إذا كنت تريد عرض الصورة المقصوصة يمكنك استخدام الكود التالي:
#plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#plt.title('Cropped Image')
#plt.show()
######################################################################################



app= Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html', appName="Intel Image Classification")


@app.route('/predictApi',methods=['POST'])
def api():
    try:
        if 'fileup' not in request.files:
            return 'please try again, no image found'
        image = request.files.get('fileup')
        image=cv2.imread(image)
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
        image = cv2.imread(image)
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

