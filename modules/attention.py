import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### 완성시켜야 할 빈 코드 블록
    # query: [bs, heads, q_len, d]
    # key:   [bs, heads, k_len, d]
    # value: [bs, heads, k_len, d]
    # attention_mask: [bs, 1, 1, k_len]

    (bs, h, seq_len, d) = key.size()
    assert (bs, h, seq_len, d)  == query.size()
    assert (bs, h, seq_len, d)  == value.size()
    assert attention_mask.size() == (bs, 1, 1, seq_len)
    # Calculate the attention scores.
    attention = query @ key.transpose(-1, -2) # [bs, num_attention_heads, seq_len, seq_len]
    attention = attention / (d ** 0.5)
    # Apply the attention mask.
    attention_mask_triu = torch.triu(torch.ones(seq_len, seq_len, device=key.device), diagonal=1)[None, None, :, :] * -10000.0
    causal_attention_mask = attention_mask + attention_mask_triu
    causal_attention_mask = torch.clamp(causal_attention_mask, min=-10000.0)
    attention = attention + causal_attention_mask
    attention = torch.nn.functional.softmax(attention, dim=-1)
    attention = self.dropout(attention)
    attention = attention @ value # [bs, num_attention_heads, seq_len, attention_head_size]
    attention = rearrange(attention, 'b h t d -> b t (h d)') # [bs, seq_len, num_attention_heads * attention_head_size]
    return attention


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
