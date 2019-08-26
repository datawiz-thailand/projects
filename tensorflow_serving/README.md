# TensorFlow Serving
> Tutorial นี้เป็นส่วนหนึ่งของ Medium blog post นี้ครับ https://medium.com/@poom.wettayakorn/https-medium-com-poom-wettayakorn-deploy-image-recognition-using-tensorflow-serving-253f210f982e

TensorFlow Serving คือ High Performance Serving System สำหรับ Machine Learningที่ออกแบบมาเพื่อใช้งานบน Production

ซึ่งเป็นตัวจัดการโมเดล inference ของ machine learning ที่ช่วยในการทำงานบน scale ที่ใหญ่ขึ้น โดยมี features ที่สามารถใช้ได้กับ use cases เหล่านี้:

* ต้องการรันหลายโมเดล หรือหลายโมเดลเวอร์ชั่นพร้อมกัน (ช่วนในการทำ experiments และ A/B testing)

* ปล่อย API endpoints ให้ฝั่ง client (ได้ทั้ง RPC และ HTTP protocol)

* ลด latency และการจัดการทรัพยากร GPU อย่างมีประสิทธิภาพ

## Serve a Tensorflow model in 5 minutes

### ResNet
เริ่มจากสร้าง folder ไว้เก็บโมเดล /tmp/resnet จากนั้นดาวน์โหลด resnet_v2_fp32_savedmodel ด้วย cURL และ extract ไฟล์

```bash
$ mkdir /tmp/resnet
$ curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz
```

### TensorFlow Serving with Docker
เมื่อได้ Pre-trained ResNet มาแล้ว ต่อไปคือรัน TensorFlow Serving server โดยวิธีที่ง่ายที่สุดคือใช้ผ่าน docker ซึ่งเพียงแค่รันไม่กี่ commands ก็ได้ทั้ง server และ API endpoints ให้เรียกใช้ได้ทันที

```bash
$ docker pull tensorflow/serving
$ docker run --rm -it -p 8501:8501 -v /tmp/resnet:/models/resnet -e MODEL_NAME=resnet tensorflow/serving
```

Docker run command นี้ประกอบไปด้วยอะไรบ้าง:

* `-p 8501:8501` : เปิดพอร์ต 8501 สำหรับ REST API (8500 สำหรับ gRPC)
* `-v /tmp/resnet:/models/resnet` : mount directory ของ local (/tmp/resnet) ไปที่ container ในชื่อ /models/resnet
* `-e MODEL_NAME=resnet` : กำหนดให้โหลดโมเดลชื่อ "resnet"

### Model Inference

ขั้นตอนสุดท้ายคือการทำโมเดล Inference โดยรัน client ในการเรียก REST API ไปที่ Serving server

```bash
$ curl -o /tmp/resnet/resnet_client.py https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
$ python /tmp/resnet/resnet_client.py
```
เท่านี้เราก็ได้ผลลัพธ์ prediction และค่า average latency อีกด้วย ซึ่งอยู่ที่ประมาณ 59 ms 👍

## Reference
* [https://github.com/tensorflow/serving](https://github.com/tensorflow/serving)
* [https://www.tensorflow.org/serving](https://www.tensorflow.org/serving)
