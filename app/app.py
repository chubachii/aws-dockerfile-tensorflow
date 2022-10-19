import boto3
import cv2
import numpy as np
import tensorflow as tf
from pymediainfo import MediaInfo

s3_client = boto3.client('s3')
#model = tf.keras.models.load_model('./model.h5', compile=False)

VIDEO_FPS = 10
MODEL_PATH = 'models/efficientdet_d0_coco17_tpu-32/saved_model/'
LABEL_PATH = 'models/coco-labels-paper.txt'
OUTPUT_WIDTH = 300

# ロード
detection_graph = tf.Graph()
loaded = tf.saved_model.load(MODEL_PATH)
labels = ['blank']
with open(LABEL_PATH,'r') as f:
    for line in f: labels.append(line.rstrip())

colors = [
  (0, 0, 255),
  (0, 64, 255),
  (0, 128, 255),
  (0, 192, 255),
  (0, 255, 255),
  (0, 255, 192),
  (0, 255, 128),
  (0, 255, 64),
  (0, 255, 0),
  (64, 255, 0),
  (128, 255, 0),
  (192, 255, 0),
  (255, 255, 0),
  (255, 192, 0),
  (255, 128, 0),
  (255, 64, 0),
  (255, 0, 0),
  (255, 0, 64),
  (255, 0, 128),
  (255, 0, 192),
  (255, 0, 255),
  (192, 0, 255),
  (128, 0, 255),
  (64, 0, 255),
]



def handler(event, context):
    try:
        for record in event['Records']:
            # アップロードされた画像をダウンロード
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            download_path = '/tmp/target_image'
            s3_client.download_file(bucket, key, download_path)
            splitkey = key.split('/')	#keyのフォルダ名/ファイル名を分離
            print("name:"+splitkey[1])

            # 読み込み用
            cap = cv2.VideoCapture(download_path)

            
            degree = 0
            if splitkey[1].endswith('.mov') or splitkey[1].endswith('.MOV') or splitkey[1].endswith('.mp4') or splitkey[1].endswith('.MP4'): 
                
                media_info = MediaInfo.parse(download_path)

                # 入力ファイルのサイズ
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                for track in media_info.tracks:
                    if track.rotation == "90.000":
                        degree = 180
                        frame_width, frame_height = w, h  
                        break      
                    elif track.rotation == "180.000":
                        degree = 0
                        frame_width, frame_height = w, h  
                        break
                    elif track.rotation == "270.000":
                        degree = 180
                        frame_width, frame_height = w, h  
                        break
                    else:
                        degree = 0
                        frame_width, frame_height = w, h

            print("rotated " + str(degree))
            

            if splitkey[1].endswith('.mp4') or splitkey[1].endswith('.MP4'):
                # 書き込み用
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
                writer = cv2.VideoWriter("/tmp/" + splitkey[1], fmt, VIDEO_FPS, (frame_width, frame_height))

                thresh = original_fps / VIDEO_FPS

            if splitkey[1].endswith('.mov') or splitkey[1].endswith('.MOV'):
                # 書き込み用
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                fmt = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter("/tmp/" + splitkey[1], fmt, VIDEO_FPS, (frame_width, frame_height))

                thresh = original_fps / VIDEO_FPS


            frame_counter = 0


            while True:
        
                ret, frame = cap.read()
                if not ret: break
                
                # 画像の入力に対応するため初回のみ1フレームずらす
                if frame_counter == 0 or frame_counter >= thresh:

                    print("loaded!")

                    # movの場合 180度回転
                    if degree == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif degree == 180: 
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif degree == 270: 
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


                    frame_height, frame_width, _ = frame.shape
                
                    # 入力画像を正方形で切り出し
                    crop_len = min(frame_height, frame_width)
                    crop_start = ((frame_height-crop_len) // 2, (frame_width-crop_len) // 2)
                    img_crp = frame[crop_start[0]:crop_start[0]+crop_len, crop_start[1]:crop_start[1]+crop_len]

                    img_bgr = cv2.resize(img_crp, (300, 300))
                    img_np  = img_bgr[:, :, ::-1]
                    img_np  = np.expand_dims(img_np, axis=0)

                    img_tensor = tf.convert_to_tensor(img_np)
                    output_dict = loaded.signatures["serving_default"](img_tensor)

                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].numpy()
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].numpy()
                    output_dict['detection_scores'] = output_dict['detection_scores'][0].numpy()

                    for i in range(output_dict['num_detections']):
                        class_id = output_dict['detection_classes'][i].astype(np.int)
                        if class_id < len(labels):
                            label = labels[class_id]
                        else:
                            label = 'unknown'

                        detection_score = output_dict['detection_scores'][i]

                        if detection_score > 0.5:

                            h, w, _ = img_crp.shape
                    
                            box = output_dict['detection_boxes'][i] * np.array([h, w,  h, w])
                            box = box.astype(np.int)

                            class_id = class_id % len(colors)
                            color = colors[class_id]

                            cv2.rectangle(frame, (box[1]+crop_start[1], box[0]+crop_start[0]), (box[3]+crop_start[1], box[2]+crop_start[0]), color, 3)

                            information = '%s: %.1f%%' % (label, output_dict['detection_scores'][i] * 100.0)
                            cv2.putText(frame, information, (box[1]+crop_start[1] + 15, box[2]+crop_start[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, int(1*crop_len/300), (255, 255, 255), int(2*crop_len/300), cv2.LINE_AA)
                            cv2.putText(frame, information, (box[1]+crop_start[1] + 15, box[2]+crop_start[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, int(1*crop_len/300), color, int(1*crop_len/300), cv2.LINE_AA)

                    if (splitkey[1].endswith('.jpg') or splitkey[1].endswith('.JPG')):
                        cv2.imwrite("/tmp/" + splitkey[1], frame)

                    elif (splitkey[1].endswith('.jpeg') or splitkey[1].endswith('.JPEG')):
                        cv2.imwrite("/tmp/" + splitkey[1], frame)

                    elif (splitkey[1].endswith('.png') or splitkey[1].endswith('.PNG')):
                        cv2.imwrite("/tmp/" + splitkey[1], frame)

                    elif (splitkey[1].endswith('.mp4') or splitkey[1].endswith('.MP4')):
                        writer.write(frame)

                    elif (splitkey[1].endswith('.MOV') or splitkey[1].endswith('.MOV')):
                        writer.write(frame)

                    frame_counter = 1
                
                else:
                    #print("skipped!")
                    frame_counter += 1


            print("done!")
            cap.release()

            if (splitkey[1].endswith('.mp4') or splitkey[1].endswith('.MP4')):
                writer.release()
            
            if (splitkey[1].endswith('.mov') or splitkey[1].endswith('.MOV')):
                writer.release()


            # 画像をアップロード
            newkey = key.replace(splitkey[0], 'result')	# result/ファイル名 のnewkeyを作成
            s3_client.upload_file("/tmp/" + splitkey[1], bucket, newkey)
            print("saved")



        return 'Success'

    except Exception as e:
        print(e)
        return 'Failure'