from flask import Flask
import cv2
import uuid
import base64
import requests
import numpy as np
from keras.models import model_from_json
from flask import request, jsonify
import json

FACE_CASCADE = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
emotions = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# load json and create model arch
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model.h5')
model._make_predict_function()

app = Flask(__name__)


def base64_encode_image(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ret, image_buf = cv2.imencode('.jpg', image_bgr, (cv2.IMWRITE_JPEG_QUALITY, 40))
    image_str = "".join(chr(x) for x in base64.b64encode(image_buf))
    return 'data:image/jpeg;base64,' + image_str


def predict_emotion(face_image_gray, index):  # a single cropped face
    resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return angry, fear, happy, sad, surprise, neutral


def build_PhotoInfo(image_gray, image_rgb, annotated_rgb, crop_faces):
    photoinfo = {}
    # this function returns coordinates of faces in grayscale
    faces = FACE_CASCADE.detectMultiScale(image_gray,
                                          scaleFactor=1.1,
                                          minNeighbors=3,
                                          minSize=(45, 45),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    face_color = [0, 0, 255]  # blue
    thickness = 4
    FaceInfo = []
    index = 0
    for x_face, y_face, w_face, h_face in faces:

        faceinfo = {'index': index}
        faceinfo['location_xy'] = (int(x_face), int(y_face))
        faceinfo['width'] = int(w_face)
        faceinfo['height'] = int(h_face)

        face_image_gray = image_gray[y_face: y_face + h_face,
                          x_face: x_face + w_face]
        angry, fear, happy, sad, surprise, neutral = predict_emotion(face_image_gray, index)
        faceinfo['prediction'] = {'angry':float(angry),
                                  'fear':float(fear),
                                  'happy':float(happy),
                                  'sad':float(sad),
                                  'surprise':float(surprise),
                                  'neutral':float(neutral)
                                  }


        if crop_faces != None:
            face_image_rgb = image_rgb[y_face: y_face + h_face,
                             x_face: x_face + w_face]
            resized_image = cv2.resize(face_image_rgb, (40, 50))
            faceinfo['thumbnail'] = base64_encode_image(resized_image)

        if annotated_rgb != None:  # opencv drawing the box
            cv2.rectangle(annotated_rgb, (x_face, y_face),
                          (x_face + w_face, y_face + h_face),
                          face_color, thickness)
        FaceInfo.append(faceinfo)
        index += 1

    photoinfo['faces'] = FaceInfo
    if annotated_rgb != None:
        photoinfo['annotated_image'] = base64_encode_image(annotated_rgb)
    # TO-DO: FINISH mongoDB
    # insert in mongoDB and return id

    photoinfo['pic_id'] = str(uuid.uuid1())
    # print 'picture_id: ', photoinfo['pic_id']
    # _id = collection.insert_one(photoinfo).inserted_id
    # mongo automatically insersts _id into photoinfo
    return faceinfo['prediction']

def likeornot(emoji):
    print(emoji['happy'])
    Positive = float(emoji['happy']) + float(emoji['surprise'])/2
    Negative = float(emoji['angry']) + float(emoji['fear']) + float(emoji['sad'])/3


    if Positive > Negative :
        return {"score":emoji,"emoji":"Positive"}
    return {"score":emoji,"emoji":"Negative"}
def obtain_images(encoded_image_str):
    if encoded_image_str == '':
        raise Error(5724, 'You must supply a non-empty input image')

    encoded_image_buf = np.fromstring(encoded_image_str, dtype=np.uint8)
    decoded_image_bgr = cv2.imdecode(encoded_image_buf, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(decoded_image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    annotate_image = (request.args.get('annotate_image', 'false').lower() == 'true')
    if annotate_image:
        annotated_rgb = np.copy(image_rgb)
    else:
        annotated_rgb = None
    crop_image = (request.args.get('crop_image', 'false').lower() == 'true')
    if crop_image:
        crop_faces = True
    else:
        crop_faces = False
    return image_rgb, image_gray, annotated_rgb, crop_faces


def obtain_feedback(request):
    feedback = {}
    if 'image_id' in request.args:
        feedback['id'] = request.args['image_id']
    else:
        raise Error(2873, 'No `image_id` provided')

    if 'face_index' in request.args:
        feedback['face_index'] = request.args['face_index']

    if 'feedback' in request.args:
        if request.args['feedback'] in emotions:
            feedback['feedback'] = request.args['feedback']
        else:
            raise Error(2873, 'Invalid `feedback` parameter')

    # insert = collection.update({"pic_id": feedback['id'], "faces.index": int(feedback['face_index'])},
    #                            {"$push": {"face.index.$.feedback": feedback['feedback']}})
    # print "INSERT STATUS: ", insert
    return feedback


@app.route('/predict', methods=['POST'])
def predict():
    '''
    Find faces and predict emotions in a photo
    找到面孔並預測照片中的情緒
    Find faces and their emotions, and provide an annotated image and thumbnails of predicted faces.
    查找面孔及其情緒，並提供帶註釋的圖像和預測面部的縮略圖。
    ---
    tags:
      - v1.0.0

    responses:
      200:
        description: A photo info objects  照片信息對象
        schema:
          $ref: '#/definitions/PhotoInfo'
      default:
        description: Unexpected error  意外錯誤
        schema:
          $ref: '#/definitions/Error'

    parameters:
      - name: image_base64
        in: query
        description: A base64 string from an image taken via webcam or photo upload. This field must be specified, you must pass an image via the `image_base64` form parameter.
        通過網絡攝像頭或照片上傳拍攝的圖像中的base64字符串。必須指定此字段，您必須通過`image_base64`表單參數傳遞圖像。
        required: false
        type: string
      - name: image_url
        in: query
        description: The URL of an image that should be processed. If this field is not specified, you must pass an image via the `image_url` form parameter.
        應處理的圖像的URL。如果未指定此字段，則必須通過`image_url`表單參數傳遞圖像。
        required: false
        type: string
      - name: image_buf
        in: formData
        description: An image that should be processed. This is used when you need to upload an image for processing rather than specifying the URL of an existing image. If this field is not specified, you must pass an image URL via the `image_buf` parameter
        應該處理的圖像。當您需要上傳圖像進行處理而不是指定現有圖像的URL時，可以使用此選項。如果未指定此字段，則必須通過`image_buf`參數傳遞圖像URL
        required: false
        type: file
      - name: annotate_image
        in: query
        description: A boolean input flag (default=false) indicating whether or not to build and return annotated images within the `annotated_image` field of each response object
        一個布爾輸入標誌（default = false），指示是否在每個響應對象的`annotated_image`字段中構建和返回帶註釋的圖像
        required: false
        type: boolean
      - name: crop_image
        in: query
        description: A boolean input flag (default=false) indicating whether or not to crop and return faces within the `thumbnails` field of each response object
        一個布爾輸入標誌（默認值= false），指示是否在每個響應對象的“縮略圖”字段中裁剪和返回面
        required: false
        type: boolean

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: PhotoInfo
          type: object
          required:
            - faces
          properties:
            id:
                type: string
                format: byte
                description: an identification number for received image  接收圖像的標識號
            faces:
              schema:
                type: array
                description: an array of emotion probabilites, face location (x,y), cropped height and width, an empty feedback form, and a base64 encoded cropped thumbnail for each face found in this image
                一個情感概率數組，面部位置（x，y），裁剪的高度和寬度，一個空的反饋表單，以及此圖像中找到的每個面部的base64編碼裁剪縮略圖
            annotated_image:
              type: string
              format: byte
              description: a base64 encoded annotated image base64編碼的帶註
    '''
    image_rgb, image_gray, annotated_rgb, crop_faces = obtain_images(request)
    photoinfo = build_PhotoInfo(image_gray, image_rgb, annotated_rgb, crop_faces)
    # photoinfo['_id'] = str(photoinfo['_id']) # makes ObjectId jsonify
    response = jsonify(photoinfo)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/emoji', methods=['POST'])
def upload_file():
    image = request.files['image']
    image_str = base64.b64encode(image.read())
    encoded_image_str = base64.b64decode(image_str)
    image_rgb, image_gray, annotated_rgb, crop_faces = obtain_images(encoded_image_str)
    photoinfo = build_PhotoInfo(image_gray, image_rgb, annotated_rgb, crop_faces)
    response = jsonify(likeornot(photoinfo))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/hello', methods=['GET'])
def Hello():
    return 'Hello ^^'

class Error(Exception):
    '''
    To return an error to the client, simply 'raise' an instance of this class at
    any point while you are processing the request. Flask is configured to 'catch'
    the raised exception and turn it into a proper error response to the client.
    '''

    def __init__(self, code, message, http_status=400):
        self.code = code
        self.message = message
        self.http_status = http_status

    def response(self):
        error_dict = {'code': self.code, 'message': self.message}
        response = jsonify(error_dict)
        response.status_code = self.http_status
        return response


@app.errorhandler(Error)
def error_raised(error):
    return error.response()


@app.errorhandler(404)
def not_found(_):
    return Error(404, 'Resource not found', 404).response()


if __name__ == '__main__':
    app.run()
