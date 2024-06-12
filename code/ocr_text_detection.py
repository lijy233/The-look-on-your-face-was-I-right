import os
import base64
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *

def detect_text_in_image(image_path, ak, sk):
    credentials = BasicCredentials(ak, sk)
    client = OcrClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(OcrRegion.value_of("cn-north-4")) \
        .build()

    try:
        # 读取本地文件并将其编码为base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        request = RecognizeWebImageRequest()
        request.body = WebImageRequestBody(
            image=image_base64
        )
        response = client.recognize_web_image(request)
        return response.result.words_block_list
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)
        return None

if __name__ == "__main__":
    ak = "EPQ2BDPM3AWRYOEBPVMJ"
    sk = "XEfYiba9yFUXLMc27aLI86MnVM9krZuLPoXcLvDT"
    img_path = "../img/1.jpg"

    words_block_list = detect_text_in_image(img_path, ak, sk)
    if words_block_list:
        print("Detected text blocks:")
        for block in words_block_list:
            print(block)
    else:
        print("No text detected or an error occurred.")
