import torch
from torch import nn
import math 
import torch.nn.functional as F

class SpatialKeyValue(nn.Module):
    
    def __init__(self,dim_in = 32, dim_out = 32):
        super(SpatialKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x))
        
class MaskedSelfAttention(nn.Module):
    # def __init__(self,args,raw_rgb_dim = 0):
    def __init__(self,args, input_dim, output_dim):
        super(MaskedSelfAttention, self).__init__()
        self.args = args

        self.qeury_embed = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1)
        self.key_embed = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1)
                                     
    def forward(self,pixel_feat, mask = None):

        key_embed = self.key_embed(pixel_feat)
        query_key = self.qeury_embed(pixel_feat)
        value_embed = self.value_embed(pixel_feat)

        k_emb = key_embed.size(1)
        A = torch.bmm(key_embed.transpose(1, 2).contiguous(), query_key)
        A = A / math.sqrt(k_emb)

        if not mask == None:
            A[mask.transpose(1, 2).squeeze(-1) == 0] -= 10

        A = F.softmax(A, dim=1)
        out = torch.bmm(value_embed, A)
        return out