# coding : utf-8
# Author : yuxiang Zeng
# 问题2: 图像检索文本
import os
import torch
from PIL import Image
import numpy as np
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from utils.config import get_config

"""
Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
Loading vision model config from /Users/zengyuxiang/Documents/实战代码/4.1 泰迪杯实战/Chinese_CLIP/cn_clip/clip/model_configs/ViT-B-16.json
Loading text model config from /Users/zengyuxiang/Documents/实战代码/4.1 泰迪杯实战/Chinese_CLIP/cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json
Model info {'embed_dim': 512, 'image_resolution': 224, 'vision_layers': 12, 'vision_width': 768, 'vision_patch_size': 16, 'vocab_size': 21128, 
            'text_attention_probs_dropout_prob': 0.1, 'text_hidden_act': 'gelu', 'text_hidden_dropout_prob': 0.1, 'text_hidden_size': 768, 'text_initializer_range': 0.02, 
            'text_intermediate_size': 3072, 'text_max_position_embeddings': 512, 'text_num_attention_heads': 12, 'text_num_hidden_layers': 12, 'text_type_vocab_size': 2}
"""
class Raw2Vector:
    def __init__(self, image_model, text_model, args):
        self.args = args
        print("Available models:", available_models())
        # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name("ViT-B-16", device=device, download_root='../BigDataSource/')
        self.model.eval()

    def image2tensor(self, image):
        image = image.unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def text2tensor(self, text):
        # text = text.unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def retrieve(self, query, database):
        # print(query.shape, database.shape)
        # image, text : torch.Size([1, 3, 224, 224]) torch.Size([4, 52])
        logits_per_image, logits_per_text = self.model.get_similarity(query, database)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        return probs


if __name__ == '__main__':
    args = get_config()
    transfer = Raw2Vector('ViT-B-16', '1', args)

    address = os.path.dirname(os.path.abspath(__file__)) + '/test_image/img_2.png'
    # print(address)

    # Get the image
    raw_image = Image.open(address)
    image_tensor = transfer.preprocess(raw_image)
    image_features = transfer.image2tensor(image_tensor)

    # Get the text
    raw_text = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "天空"]
    text_tensor = clip.tokenize(raw_text).to(args.device)
    text_features = transfer.text2tensor(text_tensor)
    # print(text_features.shape)

    # Retrieve
    probs = transfer.retrieve(image_features, text_features)
    print(probs)

    max_index = np.argmax(probs, axis=1)
    print(f"AI预测最有可能是 Text[{max_index[0]}]:{raw_text[max_index[0]]}", )
