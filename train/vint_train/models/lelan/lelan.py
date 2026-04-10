import os
import argparse
import time
import pdb

import torch
import torch.nn as nn

class LeLaN_clip(nn.Module):
    """单帧版 LeLaN 的轻量运行时封装。

    这个项目把 forward 设计成字符串分发接口，
    这样训练和部署代码可以分别调用文本编码器、视觉编码器和控制头。
    """

    def __init__(self, vision_encoder, 
                       #noise_pred_net,
                       dist_pred_net,
                       text_encoder):
        super(LeLaN_clip, self).__init__()


        self.vision_encoder = vision_encoder   
        self.text_encoder = text_encoder          
        #self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

    def eval_text_encoder(self,):
        # CLIP 文本编码器通常被冻结，所以即使训练时也保持 eval 模式。
        self.text_encoder.eval()
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            # 用 prompt 特征对当前观测图像做条件编码。
            output = self.vision_encoder(kwargs["obs_img"], kwargs["feat_text"])      
        elif func_name == "text_encoder":
            # 文本编码直接交给 CLIP。
            output = self.text_encoder.encode_text(kwargs["inst_ref"])   
        #elif func_name == "noise_pred_net":
        #    output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class LeLaN_clip_temp(nn.Module):
    """带短历史图像的时序版 LeLaN 运行时封装。"""

    def __init__(self, vision_encoder, 
                       #noise_pred_net,
                       dist_pred_net,
                       text_encoder):
        super(LeLaN_clip_temp, self).__init__()


        self.vision_encoder = vision_encoder   
        self.text_encoder = text_encoder          
        #self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

    def eval_text_encoder(self,):
        self.text_encoder.eval()
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            # 把短历史观测、当前帧和 prompt 特征编码成一个策略条件向量。
            output = self.vision_encoder(kwargs["obs_img"], kwargs["feat_text"], kwargs["current_img"])      
        elif func_name == "text_encoder":
            output = self.text_encoder.encode_text(kwargs["inst_ref"])   
        #elif func_name == "noise_pred_net":
        #    output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class DenseNetwork_lelan(nn.Module):
    """把融合后的 LeLaN 特征映射成一小段未来速度序列。"""
    def __init__(self, embedding_dim, control_horizon):
        super(DenseNetwork_lelan, self).__init__()
        
        self.max_linvel = 0.5
        self.max_angvel = 1.0
        self.control_horizon = control_horizon
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            #nn.Linear(self.embedding_dim//16, 1)
            nn.Linear(self.embedding_dim//16, 2*self.control_horizon),       
            nn.Sigmoid()     
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        # 最后的 sigmoid 先把输出限制在 [0, 1]，再线性映射到实际可执行的速度范围。
        linear_vel = self.max_linvel*output[:, 0:self.control_horizon]  #max +0.5 m/s min 0.0 m/s
        angular_vel = self.max_angvel*2.0*(output[:, self.control_horizon:2*self.control_horizon] - 0.5)  #max +1.0 rad/s min -1.0 rad/s
        #return output
        return linear_vel, angular_vel

