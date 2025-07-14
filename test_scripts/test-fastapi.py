import requests

# MARK: - LOCAL
url = "http://0.0.0.0:8080/api/inference"

response = requests.post(url, json={"sex": "male", "age": 25, "pclass": 3})

print(response.status_code)
print(response.json())

# MARK: - Wrangler
url = "http://localhost:8787/api/inference"
response = requests.post(url, json={"sex": "male", "age": 25, "pclass": 3})
print(response.status_code)
print(response.json())

# MARK: - DEPLOYED
url = "https://containers-demo.cloudflare-ds-data.workers.dev/api/inference"
response = requests.post(url, json={"sex": "male", "age": 25, "pclass": 3})
print(response.status_code)
print(response.json())









