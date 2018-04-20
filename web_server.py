from keras.preprocessing.image import img_to_array
from PIL import Image
from app import app
import numpy as np
import settings
import helpers
import flask
import redis
import uuid
import time
import json
import io

# 连接到redis数据库
db = redis.StrictRedis(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)


@app.route("/")
def homepage():
    return "Welcome to the OCR REST API!"


@app.route("/doc")
def doc():
    return "Welcome to ocr rest api documents!"


@app.route("/OCR", methods=["POST"])
def OCR():
    # 初始化最终输出数据字典
    data = {"success": False}

    # 通过POST请求获取图像数据
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # 读取图片并做预处理
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # 为图像生成对应的id，并将图像id、图像以及图像尺寸加入到队列中
            k = str(uuid.uuid4())
            d = {
                "id": k,
                "image": helpers.base64_encode_image(image),
                "shape": image.shape
            }
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            # 开始循环，直到模型服务器返回结果退出while循环
            while True:
                # 尝试从数据库中获取对应id的图像的ocr结果
                output = db.get(k)

                # 检查返回的结果是否为空，不为空则将结果加入到最终输出的字典中
                if output is not None:
                    output = output.decode("utf-8")
                    data["result"] = json.loads(output)

                    # 已获得结果，将该id对应的结果从数据库中删除
                    db.delete(k)
                    break

                # 一定的延时，给予模型一定的检测预测时间
                time.sleep(settings.CLIENT_SLEEP)

            data["success"] = True

    # 将数据字典以json的形式返回
    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Starting web service...")
    app.run(port=9090)
