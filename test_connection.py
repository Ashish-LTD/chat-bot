import requests

try: 
    response = requests.get("https://huggingface.co")
    print("Internet works! status code:", response.status_code)
except Exception as e:
    print("Network error", e)