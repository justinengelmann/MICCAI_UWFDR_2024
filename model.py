from torchvision import transforms as T
from torchvision.io import read_image
import torch
import os
import cv2
import timm
import joblib
from sklearn.svm import SVC
import pickle
import logging
import sklearn
import numpy as np
import torchvision
import sys

class model:
    def __init__(self):
        self.checkpoint = "final_model.pth"
        self.mname = "efficientnet_b0"
        self.resolution = (800, 768)
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")
        
        logging.warning(f'timm:{timm.__version__} torch:{torch.__version__} sklearn:{sklearn.__version__}')
        logging.warning(f'numpy:{np.__version__} cv2:{cv2.__version__} torchvision:{torchvision.__version__}')
        logging.warning(f'joblib:{joblib.__version__} ')
        logging.warning(f'Python: {sys.version_info}')
        try:
            logging.warning(f'pickle: {pickle.format_version}')
        except Exception as e:
            print(e)
        try:
            import onnxruntime as ort
            logging.warning(f'ort: {ort.__version__}')
        except Exception as e:
            logging.warning(f'onnxruntime seems unavailable: {e}')

    def load(self, dir_path):
        sd = torch.load(os.path.join(dir_path, self.checkpoint), map_location='cpu')
        try:
            self.mname = sd.pop('model_name')
        except:
            pass
            
        try:
            self.resolution = sd.pop('resolution')
        except:
            pass
        
        self.model = timm.create_model(self.mname, pretrained=True, num_classes=1).eval()
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()
        norm_means = (0.5,)
        norm_stds = (0.5,)
        self.transforms = T.Compose([
                        T.ToTensor(),
                        T.Normalize(norm_means, norm_stds),
                       ])
               
        example_img = torch.zeros(254, 200, 3).numpy()
        img = cv2.resize(example_img, self.resolution)   
        img = self.transforms(img)
        img = img.unsqueeze(0)
        
        self.model = torch.jit.trace(self.model, img)
        self.model = torch.jit.optimize_for_inference(self.model)
        
    @torch.inference_mode()
    def predict(self, input_image):
        # Original image
        img_orig = cv2.resize(input_image, self.resolution)
        img_orig = self.transforms(img_orig)
        
        # Flipped image
        img_flipped = torch.flip(img_orig, [2])
        
        # Brightness increased
        img_bright = cv2.resize(input_image, self.resolution)
        img_bright = np.clip(img_bright.astype(np.float32) + 0.02 * 255, 0, 255).astype(np.uint8)
        img_bright = self.transforms(img_bright)
        
        # Scaled image
        h, w = input_image.shape[:2]
        scaled_h, scaled_w = int(h * 0.95), int(w * 0.95)
        img_scaled = cv2.resize(input_image, (scaled_w, scaled_h))
        img_scaled = cv2.resize(img_scaled, self.resolution)
        img_scaled = self.transforms(img_scaled)
        
        # Use jit.fork and jit.wait for parallel execution
        future1 = torch.jit.fork(self._predict_single, img_orig)
        future2 = torch.jit.fork(self._predict_single, img_flipped)
        future3 = torch.jit.fork(self._predict_single, img_bright)
        future4 = torch.jit.fork(self._predict_single, img_scaled)
        
        pred1 = torch.jit.wait(future1)
        pred2 = torch.jit.wait(future2)
        pred3 = torch.jit.wait(future3)
        pred4 = torch.jit.wait(future4)
        
        return (pred1 + pred2 + pred3 + pred4) / 4
        
    def _predict_single(self, img):
        img = img.unsqueeze(0)
        with torch.no_grad():
            preds = self.model(img)[0]
        return float(preds)
      
