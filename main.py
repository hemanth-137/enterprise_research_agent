import requests

url = "http://localhost:8000/query"

payload = {
    "query": "What is the IP address of Margaret Miller?"
}

with requests.post(url, json=payload, stream=True) as response:
    response.raise_for_status()

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            print(chunk, end="", flush=True)

print()