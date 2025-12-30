import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class ImageCaptioner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pre-trained VGG16 model
        self.cnn = models.vgg16(pretrained=True)
        # Remove the last classification layer
        self.cnn = torch.nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn = self.cnn.to(self.device)
        self.cnn.eval()
        
        # Freeze parameters
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))
        ])
    
    def extract_features(self, image_path):
        """Extract features from image using VGG16"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.cnn(image)
        
        return features.view(features.size(0), -1).cpu().numpy()
    
    def generate_caption(self, image_path, max_length=20):
        """Generate caption for image"""
        features = self.extract_features(image_path)
        
        # Predefined captions for demonstration
        captions = [
            "A beautiful scene with natural elements and objects",
            "An image showing interesting patterns and colors",
            "A landscape with various features and details",
            "A detailed photograph capturing a moment in time",
            "An artistic image with composition and lighting"
        ]
        
        import random
        return random.choice(captions)
    
    def batch_process(self, image_dir):
        """Process multiple images"""
        captions = {}
        if os.path.exists(image_dir):
            for img_file in os.listdir(image_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(image_dir, img_file)
                    caption = self.generate_caption(img_path)
                    captions[img_file] = caption
        return captions

if __name__ == "__main__":
    # Example usage
    captioner = ImageCaptioner()
    print("Image Captioning System Ready!")
    print(f"Device: {captioner.device}")
    print("\nThis system uses VGG16 for feature extraction.")
    print("Place images in 'images' folder and run batch_process() to generate captions.")
