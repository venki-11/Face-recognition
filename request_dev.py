import requests

# Define the data to be sent in the request
data = {
    "file_urls": ["https://test-bucket-c.s3.ap-south-1.amazonaws.com/7c56c155-68c6-4b73-a6b2-3728b4a6c204","https://test-bucket-c.s3.ap-south-1.amazonaws.com/77afced2-e938-4de8-a579-d159bc27b151"]
}

# Send the POST request to the FastAPI application
response = requests.post("https://0104-106-51-0-15.ngrok.io/predict", json=data)
print(response.json())