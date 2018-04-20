# coding=utf-8
from __future__ import print_function
import numpy as np
import settings
import helpers
import redis
import time
import json
import ocr

# 连接到Redis数据库
db = redis.StrictRedis(
    host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)


def ocr_process():
    # 载入ocr模型
    model = ocr.Ocr(text_process=False)
    print("load model done")
    while True:
        # 尝试从redis数据库取出一队图像数据
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)
        # 用于放置每幅图像的id
        imageIDs = []
        batch = None

        # 遍历图像数据队列
        for q in queue:
            # 解码图像数据
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(
                q["image"], settings.IMAGE_DTYPE, q["shape"])

            # 检查数据批是否为空
            if batch is None:
                batch = image

            # 不为空则将数据堆叠起来，形成一批数据
            else:
                batch = np.vstack([batch, image])

            # 更新图像id列表
            imageIDs.append(q["id"])

        # 检查是否需要进行模型预测
        if len(imageIDs) > 0:
            # 处理数据块
            results = []
            print("* Batch size: {}".format(batch.shape))
            for img in batch:
                preds = model.predict(img)
                results.append(preds)

            for (imageID, resultSet) in zip(imageIDs, results):
                # 初始化预测结果
                output = []

                # 初始化文本检测结果
                boxes = []
                for box in resultSet[1]:
                    boxes.append([box[xy] for xy in range(8)])
                r = {"words": resultSet[0], "boxes": boxes}
                output.append(r)

                # 存储检测与识别结果
                db.set(imageID, json.dumps(output, ensure_ascii=False))
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

        # 延时
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    ocr_process()
