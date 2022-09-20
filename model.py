import torch
import torchvision
from torch import nn

from typing import Tuple, Dict

def load_model(device=torch.device('cpu'), path='models_weights/eff_net.pth'):
    eff_net = torchvision.models.efficientnet_b2()
    eff_net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, 3)
    )
    eff_net.load_state_dict(torch.load(path, map_location=device))
    model_transforms = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
    return eff_net, model_transforms


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time