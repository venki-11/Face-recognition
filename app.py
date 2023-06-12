from PIL import Image
import requests
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def getPercentage(value):
        value = float(value[2:].split("]")[0])
        return round((value/1)*100,2)

def preprocess_image(url):
    url=url+".jpeg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image = image.resize((160, 160))
    return image

def calculate_similarity(image_url1, image_url2):
    image1 = preprocess_image(image_url1)
    image2 = preprocess_image(image_url2)

    faces1 = mtcnn(image1)
    faces2 = mtcnn(image2)

    if len(faces1) == 0 or len(faces2) == 0:
        return 0.0  # No faces detected
    face1=faces1[0]
    face2=faces2[0]
    embeddings1 = facenet(face1.unsqueeze(0).to(device))
    embeddings2 = facenet(face2.unsqueeze(0).to(device))

    # cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    # similarity_percentage = (cosine_similarity.item() + 1) * 50  # Convert similarity to a percentage

    sim = np.dot(embeddings1.detach().numpy(), embeddings2.detach().numpy().T)
    similarity_percentage=getPercentage(str(sim))
    

    return similarity_percentage