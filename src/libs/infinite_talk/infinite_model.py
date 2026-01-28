from einops import rearrange, repeat
import torch
import torch.nn as nn
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
from comfy.ldm.wan.model import repeat_e,apply_rope1,sinusoidal_embedding_1d

def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range
 
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode='mean', attn_bias=None):
    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1) # B, H, x_seqlens, ref_seqlens

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = x_ref_attnmap.sum(-1) / (ref_target_mask.sum() + 1e-6) # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1) # B, x_seqlens, H
       
        if mode == 'mean':
            x_ref_attnmap = x_ref_attnmap.mean(-1) # B, x_seqlens
        elif mode == 'max':
            x_ref_attnmap = x_ref_attnmap.max(-1) # B, x_seqlens
        
        x_ref_attn_maps.append(x_ref_attnmap)
    
    del attn, x_ref_attn_map_source

    return torch.concat(x_ref_attn_maps, dim=0)

def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2):
    """Args:
        query (torch.tensor): B M H K
        key (torch.tensor): B M H K
        shape (tuple): (N_t, N_h, N_w)
        ref_target_masks: [B, N_h * N_w]
    """
    N_t, N_h, N_w = shape
    
    x_seqlens = N_h * N_w
    ref_k     = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape

    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)

    split_chunk = heads // split_num
    
    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_target_masks)
        x_ref_attn_maps += x_ref_attn_maps_perhead
    
    return x_ref_attn_maps / split_num

class RotaryPositionalEmbedding1D(nn.Module):
    def __init__(self,
                 head_dim,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)

class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768, 
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
        operations=None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = operations.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = operations.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = operations.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = operations.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = operations.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf)) 
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        context_tokens = self.norm(context_tokens.to(self.norm.weight.dtype)).to(context_tokens.dtype)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens

class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        operations=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = operations.Linear(dim, dim, bias=qkv_bias)
        self.proj = operations.Linear(dim, dim)
        self.kv_linear = operations.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None, enable_sp=False, kv_seq=None) -> torch.Tensor:
        N_t, N_h, N_w = shape
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))
        
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0)

        x = optimized_attention(
            q, 
            encoder_k, 
            encoder_v, 
            heads=self.num_heads, 
            skip_reshape=True
        )

        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape) 
        x = self.proj(x)

        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x

class SingleStreamMutiAttention(SingleStreamAttention):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        class_range: int = 24,
        class_interval: int = 4,
        operations=None,
    ) -> None:
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            operations=operations,
        )
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1  = (0, self.class_interval)
        self.rope_h2  = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        shape=None, 
        x_ref_attn_map=None,
        human_num=None,
    ) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        if human_num == 1:
            return super().forward(x, encoder_hidden_states, shape)

        N_t, _, _ = shape 
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t) 

        B, N, C = x.shape
        q = self.q_linear(x) 
        q_shape = (B, N, self.num_heads, self.head_dim) 
        q = q.view(q_shape).permute((0, 2, 1, 3))

        max_values = x_ref_attn_map.max(1).values[:, None, None] 
        min_values = x_ref_attn_map.min(1).values[:, None, None] 
        max_min_values = torch.cat([max_values, min_values], dim=2)

        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
        back   = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] # N 

        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        _, N_a, _ = encoder_hidden_states.shape 
        encoder_kv = self.kv_linear(encoder_hidden_states) 
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0) 
        
        per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
        per_frame[:per_frame.size(0)//2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[per_frame.size(0)//2:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame]*N_t, dim=0)
        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)
 
        x = optimized_attention(
            q, 
            encoder_k, 
            encoder_v, 
            heads=self.num_heads, 
            skip_reshape=True
        )

        # linear transform
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape) 
        x = self.proj(x) 

        # reshape x to origin shape
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t) 

        return x

@torch.compiler.disable
def infinite_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    audio_window = 5
    vae_scale = 4

    first_latent = transformer_options.get('first_latent', None)
    if first_latent is not None:
        if x.shape[3] != first_latent.shape[3] or x.shape[4] != first_latent.shape[4]:
            raise ValueError(f"Video width or height must dimunish by 16")
        x[:,:first_latent.shape[1], :first_latent.shape[2]] = first_latent

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
    e = e.reshape(t.shape[0], -1, e.shape[-1])
    e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    full_ref = None
    if self.ref_conv is not None:
        full_ref = kwargs.get("reference_latent", None)
        if full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            x = torch.concat((full_ref, x), dim=1)

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    # ============= inject infinite logic start =============
    audio_cond = transformer_options.get('audio_cond', None)
    first_frame_audio_emb_s = audio_cond[:, :1, ...] 

    latter_frame_audio_emb = audio_cond[:, 1:, ...] 
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=vae_scale) 
    middle_index = audio_window // 2

    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 

    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
    
    latter_frame_audio_emb_s = torch.concat([
        latter_first_frame_audio_emb, 
        latter_middle_frame_audio_emb, 
        latter_last_frame_audio_emb
    ], dim=2) 

    audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) 
    audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

    ref_target_masks = transformer_options.get('ref_target_masks', None)
    token_ref_target_masks = transformer_options.get('token_ref_target_masks', None)
    if ref_target_masks is not None and token_ref_target_masks is None:
        N_h, N_w = grid_sizes[1], grid_sizes[2]

        ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
        token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest') 
        token_ref_target_masks = token_ref_target_masks.squeeze(0)
        token_ref_target_masks = (token_ref_target_masks > 0)
        token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
        token_ref_target_masks = token_ref_target_masks.to(x.dtype)

        transformer_options['token_ref_target_masks'] = token_ref_target_masks

    transformer_options['grid_sizes'] = grid_sizes
    transformer_options['audio_embedding'] = audio_embedding
    # ============= inject infinite logic end =============

    transformer_options["total_blocks"] = len(self.blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.blocks):
        transformer_options["block_index"] = i
        x = block(
            x, 
            e=e0, 
            freqs=freqs, 
            context=context, 
            context_img_len=context_img_len, 
            transformer_options=transformer_options
        )

    # head
    x = self.head(x, e)

    if full_ref is not None:
        x = x[:, full_ref.shape[1]:]

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x

def infinite_block_forward(
    self,
    x,
    e,
    freqs,
    context,
    context_img_len=257,
    transformer_options={},
):
    if e.ndim < 4:
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
    else:
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

    # self-attention
    x = x.contiguous() # otherwise implicit in LayerNorm
    y = self.self_attn(
        torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
        freqs, 
        transformer_options=transformer_options
    )

    x = torch.addcmul(x, y, repeat_e(e[2], x))
    del y

    # cross-attention & ffn
    x = x + self.cross_attn(
        self.norm3(x), 
        context, 
        context_img_len=context_img_len, 
        transformer_options=transformer_options
    )

    # ============= inject infinite logic start =============
    grid_sizes = transformer_options.get('grid_sizes', None)
    x_ref_attn_map = transformer_options.get('x_ref_attn_map', None)
    audio_embedding = transformer_options.get('audio_embedding', None)
    human_num = transformer_options.get('human_num', None)
    x_a = self.audio_cross_attn(
        self.norm_x(x), 
        encoder_hidden_states=audio_embedding,
        shape=grid_sizes, 
        x_ref_attn_map=x_ref_attn_map, 
        human_num=human_num
    )
    x = x + x_a
    # ============= inject infinite logic end =============

    y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
    x = torch.addcmul(x, y, repeat_e(e[5], x))
    return x

def infinite_block_self_attn_forward(
    self, 
    x, 
    freqs, 
    transformer_options={}
):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    grid_sizes = transformer_options.get('grid_sizes', None)

    def qkv_fn_q(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        return apply_rope1(q, freqs)

    def qkv_fn_k(x):
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        return apply_rope1(k, freqs)

    #These two are VRAM hogs, so we want to do all of q computation and
    #have pytorch garbage collect the intermediates on the sub function
    #return before we touch k
    q = qkv_fn_q(x)
    k = qkv_fn_k(x)

    x = optimized_attention(
        q.view(b, s, n * d),
        k.view(b, s, n * d),
        self.v(x).view(b, s, n * d),
        heads=self.num_heads,
        transformer_options=transformer_options,
    )

    x = self.o(x)

    # =========== inject infinite logic start ============
    token_ref_target_masks = transformer_options.get('token_ref_target_masks', None)
    if token_ref_target_masks is not None and grid_sizes is not None:
        x_ref_attn_map = get_attn_map_with_target(
            q, 
            k,
            grid_sizes, 
            ref_target_masks=token_ref_target_masks
        )
        transformer_options['x_ref_attn_map'] = x_ref_attn_map
    # =========== inject infinite logic end ============

    return x