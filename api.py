from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from app import calculate_similarity
import os
import sys
from pyngrok import ngrok

class Model(BaseModel):
    file_urls: List[str]

app = FastAPI()

@app.post("/predict")
async def predict(item: Model):
    file_urls = item.file_urls
    if len(file_urls) != 2:
        return {"error": "Please provide exactly 2 image URLs"}

    image_url1 = file_urls[0]
    image_url2 = file_urls[1]

    similarity_percentage = calculate_similarity(image_url1, image_url2)

    return {"result": similarity_percentage}

if __name__=="__main__":
    ngrok_tunnel = ngrok.connect(9002)
    public_url = ngrok_tunnel.public_url
    print(f"ngrok tunnel is active. Public URL: {public_url}")
    try:
        # Start the FastAPI server
        uvicorn.run(app,port=9002,host="0.0.0.0")
    except KeyboardInterrupt:
        # Stop the ngrok tunnel when the server is stopped
        ngrok.kill()
