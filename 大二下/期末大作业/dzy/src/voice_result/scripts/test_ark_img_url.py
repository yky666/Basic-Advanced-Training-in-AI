import requests

ARK_API_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
ARK_AUTH_TOKEN = "ark-ef377b91-01ce-4f42-804c-96b753d7608e-9e43b"
MODEL_ID = "doubao-seed-2-0-pro-260215"

headers = {
    "Authorization": f"Bearer {ARK_AUTH_TOKEN}",
    "Content-Type": "application/json"
}
payload = {
    "model": MODEL_ID,
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"},
                {"type": "input_text", "text": "简短描述图片里有什么"}
            ]
        }
    ]
}
resp = requests.post(ARK_API_URL, json=payload, headers=headers, timeout=15)
print("识图接口完整返回：\n", resp.text)