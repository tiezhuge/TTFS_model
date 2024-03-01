import torch

# 定义数字嵌入张量
nums_embed = torch.randn(128, 706)
nums_embed=nums_embed.unsqueeze(2)
print(nums_embed.shape)
# 定义注意力机制
attention = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1)

# 计算注意力权重
attn_weights, _ = attention(nums_embed.transpose(0, 1), nums_embed.transpose(0, 1), nums_embed.transpose(0, 1))

# 使用注意力权重加权数字嵌入
weighted_embed = torch.matmul(attn_weights.transpose(0, 1), nums_embed)

# 打印结果
print("Weighted Embedding Shape:", weighted_embed.shape)
print("Attention Scores Shape:", attn_weights.shape)
