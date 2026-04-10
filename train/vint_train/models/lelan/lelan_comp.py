import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding

import clip


class LeLaN_clip_FiLM(nn.Module):
    """单帧版 LeLaN 编码器。

    它会先把 CLIP 文本特征变成 FiLM 参数，用这些参数调制视觉主干，
    再把得到的视觉特征压缩并送入一个轻量 transformer 继续融合。
    """
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        feature_size: int = 1024,
        clip_type: str = "ViT-B/32",
    ) -> None:
        """
        LeLaN Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        
        # 根据所选 CLIP 文本特征维度，构建对应大小的 FiLM 视觉主干。
        if clip_type == "ViT-B/32":
            self.film_model = build_film_model(8, 10, 128, 512)
        elif clip_type == "ViT-L/14@336px":
            self.film_model = build_film_model(8, 10, 128, 768) 
        elif clip_type == "RN50x64":
            self.film_model = build_film_model(8, 10, 128, 1024) 
                        
        self.num_goal_features = feature_size
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size) #clip feature
        else:
            self.compress_goal_enc = nn.Identity()

        # 虽然单帧版本质上只有一个 token，但这里仍保留浅层 transformer，
        # 这样它和时序版的接口保持一致。
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=2) #no context
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)

    def forward(self, obs_img: torch.tensor, feat_text: torch.tensor):#inst_ref: torch.tensor
        device = obs_img.device
        # 这里没有额外单独的 goal image token，
        # 而是直接用文本特征去条件化当前图像本身。
        obsgoal_img = obs_img       
        inst_encoding = feat_text
        obsgoal_encoding = self.film_model(obsgoal_img, inst_encoding)
        obsgoal_encoding_cat = obsgoal_encoding.flatten(start_dim=1)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding_cat)        

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        obs_encoding = obsgoal_encoding                
        
        # 保持与时序编码器一致的输出形状。
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens

class LeLaN_clip_FiLM_temp(nn.Module):
    """带短历史图像与 prompt 条件的时序版 LeLaN 编码器。"""
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        feature_size: int = 1024,
        clip_type: str = "ViT-B/32",
    ) -> None:
        """
        LeLaN Encoder class
        """    
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        
        # 历史帧走 EfficientNet，当前帧则通过下面的 FiLM 分支与 prompt 融合。
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        # 当前帧通过 FiLM 与 prompt 做条件融合。
        if clip_type == "ViT-B/32":
            self.film_model = build_film_model(8, 10, 128, 512)
        elif clip_type == "ViT-L/14@336px":
            self.film_model = build_film_model(8, 10, 128, 768) 
        elif clip_type == "RN50x64":
            self.film_model = build_film_model(8, 10, 128, 1024) 
                        
        self.num_goal_features = feature_size
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size) #clip feature
        else:
            self.compress_goal_enc = nn.Identity()

        # 最终由 transformer 融合：历史 token + 一个带 prompt 条件的当前帧 token。
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2) #no context
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)

    def forward(self, obs_img: torch.tensor, feat_text: torch.tensor, current_img: torch.tensor):#inst_ref: torch.tensor

        device = obs_img.device
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)
                
        # CLIP 文本特征由模块外部先算好，再传进来。
        inst_encoding = feat_text
        
        # 当前图像先被编码成一个带 prompt 条件的 token。
        obsgoal_encoding = self.film_model(current_img, inst_encoding)        
        obsgoal_encoding_cat = obsgoal_encoding.flatten(start_dim=1)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding_cat)    
        
        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding         
            
        # dataloader 会先把历史帧沿通道维拼起来，这里再拆回一张张 RGB 图分别编码。
        obs_img_list = torch.split(obs_img, 3, dim=1)        
        obs_img = torch.concat(obs_img_list, dim=0)        
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)

        obs_encoding = self.compress_obs_enc(obs_encoding)   
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)      
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        
        # transformer 最终看到的是 [历史帧..., 当前 prompt 条件 token]。
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def create_conv_layer(in_channels, out_channels, kernel_size, stride, padding):
    # 轻量 FiLM 视觉塔里复用的卷积层构造函数。
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


class InitialFeatureExtractor(nn.Module):
    """FiLM 调制前的前段空间特征提取器。"""
    def __init__(self):
        super(InitialFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(
            create_conv_layer(3, 128, 5, 2, 2),
            create_conv_layer(128, 128, 3, 2, 1),
            create_conv_layer(128, 128, 3, 2, 1),
        )
        
    def forward(self, x):
        return self.layers(x)

class IntermediateFeatureExtractor(nn.Module):
    """FiLM 残差块之后继续提特征的后段卷积塔。"""
    def __init__(self):
        super(IntermediateFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(       
            create_conv_layer(128, 256, 3, 2, 1),
            create_conv_layer(256, 512, 3, 2, 1),
            create_conv_layer(512, 1024, 3, 2, 1),
            create_conv_layer(1024, 1024, 3, 2, 1),                                
        )
        
    def forward(self, x):
        return self.layers(x)

        
class FiLMTransform(nn.Module):
    """用文本生成的 gamma/beta 做逐通道仿射调制。"""
    def __init__(self):
        super(FiLMTransform, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x
        
        
class ResidualBlock(nn.Module):
    """在归一化特征上施加 FiLM 调制的残差块。"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.film_transform = FiLMTransform()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film_transform(x, beta, gamma)
        x = self.relu2(x)
        
        x = x + identity
        
        return x

class FinalClassifier(nn.Module):
    """保留下来的原始 FiLM 分类头，这里基本不参与主流程。"""
    def __init__(self, input_channels, num_classes):
        super(FinalClassifier, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        feature_map = x
        x = self.global_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc_layers(x)
        
        return x, feature_map
        
        
class FiLMNetwork(nn.Module):
    """根据语言特征生成条件化的视觉特征。"""
    def __init__(self, num_res_blocks, num_classes, num_channels, question_dim):
        super(FiLMNetwork, self).__init__()
        question_feature_dim = question_dim

        self.film_param_generator = nn.Linear(question_feature_dim, 2 * num_res_blocks * num_channels)
        self.initial_feature_extractor = InitialFeatureExtractor()
        self.residual_blocks = nn.ModuleList()
        self.intermediate_feature_extractor = IntermediateFeatureExtractor()
        
        for _ in range(num_res_blocks):
            self.residual_blocks.append(ResidualBlock(num_channels + 2, num_channels))
            
        self.final_classifier = FinalClassifier(num_channels, num_classes)
    
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        
    def forward(self, x, question):
        batch_size = x.size(0)
        device = x.device
        
        x = self.initial_feature_extractor(x)
        # 把文本特征变成每个残差块各自的一组 (beta, gamma)。
        film_params = self.film_param_generator(question).view(
            batch_size, self.num_res_blocks, 2, self.num_channels)
        
        d = x.size(2)
        # 额外拼接坐标通道，让网络保留基本的空间位置信息。
        coords = torch.arange(-1, 1 + 0.00001, 2 / (d-1)).to(device)
        coord_x = coords.expand(batch_size, 1, d, d)
        coord_y = coords.view(d, 1).expand(batch_size, 1, d, d)
        
        for i, res_block in enumerate(self.residual_blocks):
            beta = film_params[:, i, 0, :]
            gamma = film_params[:, i, 1, :]
            
            x = torch.cat([x, coord_x, coord_y], 1)
            x = res_block(x, beta, gamma)
        
        features = self.intermediate_feature_extractor(x)
        
        return features

def build_film_model(num_res_blocks, num_classes, num_channels, question_dim):
    # 小型工厂函数，保持原训练代码的调用风格。
    return FiLMNetwork(num_res_blocks, num_classes, num_channels, question_dim)


"""
def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.model = nn.Sequential(
            conv(3, 128, 5, 2, 2),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 2, 1),

        )
        
    def forward(self, x):
        return self.model(x)

class FeatureExtractor_last(nn.Module):
    def __init__(self):
        super(FeatureExtractor_last, self).__init__()
        
        self.model = nn.Sequential(       
            conv(128, 256, 3, 2, 1),
            conv(256, 512, 3, 2, 1),
            conv(512, 1024, 3, 2, 1),
            conv(1024, 1024, 3, 2, 1),                                
        )
        
    def forward(self, x):
        return self.model(x)

        
class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        #print(gamma.size(), x.size(), beta.size())
        x = gamma * x + beta
        
        return x
        
        
class ResBlock(nn.Module):
    def __init__(self, in_place, out_place):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_place, out_place, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_place, out_place, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_place)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)
        
        x = x + identity
        
        return x

class Classifier(nn.Module):
    def __init__(self, prev_channels, n_classes):
        super(Classifier, self).__init__()
        
        self.conv = nn.Conv2d(prev_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.model = nn.Sequential(nn.Linear(512, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, n_classes))
        
    def forward(self, x):
        x = self.conv(x)
        feature = x
        x = self.global_max_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model(x)
        
        return x, feature
        
        
class FiLM(nn.Module):
    def __init__(self, n_res_blocks, n_classes, n_channels, n_dim_question):
        super(FiLM, self).__init__()
        dim_question = n_dim_question

        self.film_generator = nn.Linear(dim_question, 2 * n_res_blocks * n_channels)
        self.feature_extractor = FeatureExtractor()
        self.res_blocks = nn.ModuleList()
        self.feature_extractor_last = FeatureExtractor_last()
        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock(n_channels + 2, n_channels))
            
        self.classifier = Classifier(n_channels, n_classes)
    
        self.n_res_blocks = n_res_blocks
        self.n_channels = n_channels
        
    def forward(self, x, question):
        batch_size = x.size(0)
        device = x.device
        
        x = self.feature_extractor(x)
        film_vector = self.film_generator(question).view(
            batch_size, self.n_res_blocks, 2, self.n_channels)
        
        d = x.size(2)
        coordinate = torch.arange(-1, 1 + 0.00001, 2 / (d-1)).to(device)
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)
        
        for i, res_block in enumerate(self.res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]
            
            x = torch.cat([x, coordinate_x, coordinate_y], 1)
            x = res_block(x, beta, gamma)
        
        feature = self.feature_extractor_last(x)
        
        return feature

def make_model(n_res, n_classes, n_channels, n_dim_question):
    return FiLM(n_res, n_classes, n_channels, n_dim_question)
"""    
