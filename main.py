import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def quantize_int8(x):
    scale = x.abs().amax(dim=(-2, -1), keepdim=True) / 127
    scale = torch.clamp(scale, min=1e-8)

    q = torch.round(x / scale)
    q = q.clamp(-128, 127).to(torch.int8)

    return q, scale


def dequantize_int8(q, scale):
    return q.float() * scale


class CompressedGPT2Attention(GPT2Attention):
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, 
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(q, self.num_heads, self.head_dim)
        key   = self._split_heads(k, self.num_heads, self.head_dim)
        value = self._split_heads(v, self.num_heads, self.head_dim)

        if layer_past is not None:
            (kq, ks), (vq, vs) = layer_past

            past_key   = dequantize_int8(kq, ks)
            past_value = dequantize_int8(vq, vs)

            key   = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache:
            kq, ks = quantize_int8(key)
            vq, vs = quantize_int8(value)

            present = ((kq, ks), (vq, vs))

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = model.to(device)
model.eval()

for block in model.transformer.h:
    old_attn = block.attn

    new_attn = CompressedGPT2Attention(
        old_attn.config,
        is_cross_attention=old_attn.is_cross_attention,
        layer_idx=old_attn.layer_idx
    )

    new_attn.load_state_dict(old_attn.state_dict())

    block.attn = new_attn


prompt = "The future of artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        use_cache=True,
        do_sample=False,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))