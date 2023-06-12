from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def getPercentageExisting(value):
        value = float(value[2:].split("]")[0])
        return round((value/1)*100,2)

def getPercentage(value):
    percentage = round(((value.item() + 1) / 2) * 100, 2)
    return percentage

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512))  # Resize the image to 160x160 pixels
    return image

def calculate_similarity(embeddings1, embeddings2):
    cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    similarity_percentage = (cosine_similarity + 1) / 2 * 100
    return similarity_percentage.item()

def existingMetric(embeddings1, embeddings2):
    sim = np.dot(embeddings1.detach().numpy(), embeddings2.detach().numpy().T)
    sim1=getPercentage(sim)
    sim2=getPercentageExisting(str(sim))
    print(sim)
    print(sim1)
    print(sim2)

image1 = preprocess_image('images/img1.jpeg')
image2 = preprocess_image('images/img2.jpeg')

faces1 = mtcnn(image1)
faces2 = mtcnn(image2)

face1 = faces1[0]
face2 = faces2[0]

embeddings1 = facenet(face1.unsqueeze(0).to(device))
embeddings2 = facenet(face2.unsqueeze(0).to(device))

similarity_score = calculate_similarity(embeddings1, embeddings2)
print(f"Face similarity score: {similarity_score}")
existingMetric(embeddings1,embeddings2)

face1_np = face1.squeeze().permute(1, 2, 0).cpu().numpy()

# Convert BGR to RGB
face1_rgb = cv2.cvtColor(face1_np, cv2.COLOR_BGR2RGB)

# Display the face image
cv2.imshow("Face 1", face1_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()