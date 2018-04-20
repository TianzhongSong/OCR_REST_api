# coding=utf-8
import requests

KERAS_REST_API_URL = "http://127.0.0.1:9090/OCR"
IMAGE_PATH = "123.png"

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()

if r["success"]:
    # print r
    results = r["result"][0]

    for word in results["words"]:
        print word

    for box in results["boxes"]:
        print box

else:
    print("Request failed")
