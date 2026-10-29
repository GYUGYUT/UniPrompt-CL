import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import torch
from network.loss import *
import math
from transformers import AutoImageProcessor, AutoModel
import copy

        
class ViTWithCustomHead(nn.Module):
    def __init__(self, num_classes=5, num_layers=5, prompt_pool=None, pretrained_model_name="facebook/dinov2-base"):
        #1. facebook/dinov2-base
        #2. ClementP/FundusDRGrading-vit_base_patch14_dinov2
        #3. google/vit-base-patch16-224

        super(ViTWithCustomHead, self).__init__()

        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity() 
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        self.num_layers = num_layers

        # Classification head
        self.classification_heads = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.backbone.config.hidden_size // 2, num_classes)  # Final classification output
        )
        for layer in self.classification_heads:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Bias 

        self.prompt_pool = prompt_pool

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.encoder.layer)
        print(num_layers)
        print(self.backbone)

        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print("trainable_params :", trainable_params)
        print("total_params : ", total_params)

        self.scale = 1.0 / math.sqrt(self.backbone.config.hidden_size)
        print("self.scale : ", self.scale)

    def forward(self, x):


        if hasattr(self, 'module'):
            backbone = self.module.backbone
            prompt_pool = self.module.prompt_pool
            classification_heads = self.module.classification_heads
            num_layers = self.module.num_layers
        else:
            backbone = self.backbone
            prompt_pool = self.prompt_pool
            classification_heads = self.classification_heads
            num_layers = self.num_layers

        hidden_states = backbone.embeddings(x) 
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        prompt_losses = []
        cls_index = 0
        for l, layer in enumerate(backbone.encoder.layer):
            if l < num_layers:
 
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                cls_token_query = hidden_states[:, cls_index, :]
                
                prompt_kv, loss = prompt_pool(cls_token_query)
                prompt_losses.append(loss)
                if prompt_kv.dim() == 2:
                    prompt_kv = prompt_kv.unsqueeze(1)
                hidden_states = torch.cat([prompt_kv, hidden_states], dim=1)
                cls_index += prompt_kv.shape[1]

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = layer(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        cls_token_final = hidden_states[:, cls_index, :]
        
        logits = classification_heads(cls_token_final)

        if prompt_losses:
            prompt_loss = torch.stack(prompt_losses).mean()
        else:
            prompt_loss = logits.sum() * 0.0

        return logits , prompt_loss

    def apply_prefix_tuning_to_layer(self,layer, prompt):

        orig_key_forward = layer.attention.attention.key.forward
        orig_value_forward = layer.attention.attention.value.forward

        def key_forward_with_prefix(hidden_states):


            key_hidden = orig_key_forward(hidden_states)
            batch_size = key_hidden.size(0)
    
            prompt_expanded = prompt.unsqueeze(0).expand(batch_size, -1, -1)

            prompt_key, _ = torch.chunk(prompt_expanded, 2, dim=1)
            

            new_key = torch.cat([prompt_key, key_hidden], dim=1)
            return new_key

        def value_forward_with_prefix(hidden_states):


            value_hidden = orig_value_forward(hidden_states)
            batch_size = value_hidden.size(0)

            prompt_expanded = prompt.unsqueeze(0).expand(batch_size, -1, -1)
            _, prompt_value = torch.chunk(prompt_expanded, 2, dim=1)
            
 
            new_value = torch.cat([prompt_value, value_hidden], dim=1)
            return new_value
        layer.attention.attention.key.forward = key_forward_with_prefix
        layer.attention.attention.value.forward = value_forward_with_prefix


    def get_backbone(self):
        """Return the backbone model only."""
        return self.backbone

    def get_prompt_pool(self):
        """Return the prompt pool model only."""
        return self.prompt_pool

class PromptPool(nn.Module):
    def __init__(self, num_prompts=100, prompt_dim=768, fixed_ratio=0):

        super(PromptPool, self).__init__()

        self.num_prompts = num_prompts
        self.fixed_num = fixed_ratio


        self.keys = nn.Parameter(torch.randn(num_prompts,prompt_dim))
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim))

        nn.init.xavier_normal_(self.keys)
        nn.init.xavier_normal_(self.prompts)


        self.register_buffer("fixed_mask", torch.zeros(num_prompts, dtype=torch.bool))
        self.fixed_mask[:self.fixed_num] = True
        self.register_buffer("fixed_keys_snapshot", torch.empty(0, prompt_dim))
        self.register_buffer("fixed_prompts_snapshot", torch.empty(0, prompt_dim))
        self._freeze_hooks_registered = False

    def freeze_fixed_prompts(self):


        if self.fixed_num <= 0:
            print("No pre-fixed prompts to freeze")
            return

        fixed_num = min(self.fixed_num, self.num_prompts)
        self.fixed_mask.zero_()
        self.fixed_mask[:fixed_num] = True

        with torch.no_grad():
            self.fixed_keys_snapshot = self.keys[:fixed_num].detach().clone()
            self.fixed_prompts_snapshot = self.prompts[:fixed_num].detach().clone()

        if not self._freeze_hooks_registered:
            self.keys.register_hook(self._mask_fixed_gradients)
            self.prompts.register_hook(self._mask_fixed_gradients)
            self._freeze_hooks_registered = True
        print("✅ Pre-Fixed Prompt Freezen")

    def _mask_fixed_gradients(self, grad):
        if self.fixed_num <= 0:
            return grad
        mask = self.fixed_mask.to(grad.device).unsqueeze(-1)
        return grad.masked_fill(mask, 0)

    def restore_fixed_prompts(self):
        if self.fixed_num <= 0 or self.fixed_keys_snapshot.numel() == 0:
            return
        fixed_num = min(self.fixed_num, self.num_prompts)
        with torch.no_grad():
            self.keys[:fixed_num].copy_(self.fixed_keys_snapshot.to(self.keys.device))
            self.prompts[:fixed_num].copy_(self.fixed_prompts_snapshot.to(self.prompts.device))

    def forward(self, query):


        device = query.device

        keys = self.keys.to(device)
        prompts = self.prompts.to(device)


        prompt_weight = torch.matmul(query, keys.T)

        prompt = (prompt_weight.unsqueeze(-1) * prompts.unsqueeze(0)).sum(dim=1)

        query_norm = F.normalize(query, p=2, dim=-1)


        keys_norm = F.normalize(keys, p=2, dim=-1)


        similarity = torch.matmul(keys_norm, query_norm.T)  


        similarity = F.softmax(similarity, dim=0)  

        return prompt, self.compute_surrogate_loss(similarity)
    def compute_surrogate_loss(self, similarity, topk_indices=None):


        sim_transposed = similarity.T
        new_start = int(self.fixed_num)

        if new_start <= 0 or new_start >= self.num_prompts:
            return similarity.sum() * 0.0

        new_prompt_scores = sim_transposed[:, new_start:]


        return new_prompt_scores.mean()




