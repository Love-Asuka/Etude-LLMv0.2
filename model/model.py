import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
import tiktoken
import math

torch.manual_seed(1024)

@dataclass
class EtudeConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257
    # MOE specific parameters
    use_moe: bool = False
    expert_number: int = 4
    top_k: int = 2
    shared_experts_number: int = 2  # 虽然定义但未在SparseMOE中使用

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size

        self.register_buffer(
            'attention_mask', 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0, 
            float('-inf')
        ) / math.sqrt(self.head_size)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads], 
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)
    
    def forward(self, x):
        return self.linear(x)
    
class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        return router_logits, router_weights, selected_experts, expert_mask


class MOEConfig:
    def __init__(
            self, 
            hidden_dim, 
            expert_number, 
            top_k, 
            shared_experts_number=2,
        ):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)

        router_logits, router_weights, selected_experts, expert_mask = self.router(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * router_weights[top_x, idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits, selected_experts


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        
        # 根据配置选择使用MOE或标准FFN
        self.use_moe = config.use_moe
        if self.use_moe:
            moe_config = MOEConfig(
                hidden_dim=config.n_embd,
                expert_number=config.expert_number,
                top_k=config.top_k,
                shared_experts_number=config.shared_experts_number
            )
            self.ffn = SparseMOE(moe_config)
        else:
            self.ffn = FeedForward(config)
            
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Attention部分
        x_att = self.att(self.ln1(x))
        x = x + x_att
        
        # FFN/MOE部分
        if self.use_moe:
            x_ffn, router_logits, selected_experts = self.ffn(self.ln2(x))
            x = x + x_ffn
            return x, router_logits, selected_experts
        else:
            x_ffn = self.ffn(self.ln2(x))
            x = x + x_ffn
            return x, None, None


class Etude(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        

        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        

        self.blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            self.blocks.append(Block(config))
        

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        

        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        x = tok_emb + pos_emb
        

        total_aux_loss = 0.0
        aux_loss_coef = 0.01  # 辅助损失系数
        

        for block in self.blocks:
            x, router_logits, selected_experts = block(x)
            

            if router_logits is not None and self.config.use_moe:
                aux_loss = self.compute_aux_loss(router_logits, selected_experts)
                total_aux_loss = total_aux_loss + aux_loss
        

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        

        loss = None
        if targets is not None:
 
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            

            loss = ce_loss + aux_loss_coef * total_aux_loss
        
        return logits, loss

    def compute_aux_loss(self, router_logits, selected_experts):


        router_probs = F.softmax(router_logits, dim=-1)
        

        expert_mask = F.one_hot(selected_experts, num_classes=self.config.expert_number)
        expert_mask = expert_mask.sum(dim=1)  # (B*T, expert_number)
        

        expert_count = expert_mask.sum(dim=0)  # (expert_number,)
        

        router_prob_sum = router_probs.sum(dim=0)  # (expert_number,)
        

        loss_aux = self.config.expert_number * torch.sum(
            (expert_count / expert_count.sum()) * router_prob_sum
        )
        
        return loss_aux


def test_token_level_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)


test_token_level_moe()

class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json

        self.encoded_data = []

        self.max_lines = 1000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i+self.block_size+1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)
    



    

