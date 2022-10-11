import boto3
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (decode_predictions, preprocess_input)
from tensorflow.keras.preprocessing.image import img_to_array, load_img

s3_client = boto3.client('s3')
model = tf.keras.models.load_model('./model.h5', compile=False)

def handler(event, context):
    try:
        for record in event['Records']:
            # アップロードされた画像をダウンロード
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            download_path = '/tmp/target_image'
            s3_client.download_file(bucket, key, download_path)

            # 画像を読み込んで前処理
            img = load_img(download_path, target_size=(224, 224))
            img = img_to_array(img)
            img = img[tf.newaxis, ...]
            img = preprocess_input(img)

            # 推論
            predict = model.predict(img)
            result = decode_predictions(predict, top=5)
            print(result[0][0][1])

            # 推論結果を画像に書き込み
            width = 300
            img = cv2.imread(download_path)
            h, w = img.shape[:2]
            height = round(h * (width / w))
            img = cv2.resize(img, dsize=(width, height))
            cv2.putText(img, result[0][0][1], (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, result[0][0][1], (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imwrite('/tmp/image.jpeg', img)

            # 画像をアップロード
            s3_client.upload_file('/tmp/image.jpeg', bucket, "result.jpeg", ExtraArgs={"ContentType": "image/jpeg"})
            print("saved")


        return 'Success'

    except Exception as e:
        print(e)
        return 'Failure'