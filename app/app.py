import boto3
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (decode_predictions,
                                                        preprocess_input)
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
            print(result)

        return 'Success'

    except Exception as e:
        print(e)
        return 'Failure'