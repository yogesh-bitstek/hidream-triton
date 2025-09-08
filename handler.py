import runpod
import requests

TRITON_URL = "http://127.0.0.1:8000/v2/models/hidream/versions/1/infer"

def handler(event):
    prompt = event["input"].get("prompt", "A dreamlike painting of a futuristic city")

    request_data = {
        "inputs": [
            {
                "name": "PROMPT",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(TRITON_URL, headers=headers, json=request_data)

    if response.status_code != 200:
        return {"error": response.text}

    response_data = response.json()
    return {"image_bytes": response_data["outputs"][0]["data"]}

runpod.serverless.start({"handler": handler})
