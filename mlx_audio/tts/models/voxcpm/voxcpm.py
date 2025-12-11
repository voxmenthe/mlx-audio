import json
from pathlib import Path
from typing import Optional, List, Union, Generator, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
from transformers import AutoTokenizer

from ..base import GenerationResult
from .config import ModelArgs, LMConfig, AudioVAEConfig
from .minicpm import MiniCPMModel
from .encoder import VoxCPMLocEnc
from .dit import UnifiedCFM, VoxCPMLocDiT
from .audio_vae import AudioVAE

class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def __call__(self, x):
        x = self.in_proj(x)
        x = mx.tanh(x)
        # Rounding
        x = mx.round(x * self.scale) / self.scale
        return self.out_proj(x)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.feat_dim = args.feat_dim
        
        # LM Backbone
        self.base_lm = MiniCPMModel(args.lm_config)
        
        # Residual LM (vocab_size=0)
        res_config = LMConfig(**vars(args.lm_config))
        res_config.num_hidden_layers = args.residual_lm_num_layers
        res_config.vocab_size = 0
        self.residual_lm = MiniCPMModel(res_config)
        
        # Encoder
        enc_config = LMConfig(**vars(args.lm_config))
        enc_config.hidden_size = args.encoder_config.hidden_dim
        enc_config.intermediate_size = args.encoder_config.ffn_dim
        enc_config.num_attention_heads = args.encoder_config.num_heads
        enc_config.num_hidden_layers = args.encoder_config.num_layers
        enc_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(enc_config, input_dim=args.feat_dim)
        
        # DiT / CFM
        dit_config = LMConfig(**vars(args.lm_config)) # base on LM config but override
        dit_config.hidden_size = args.dit_config.hidden_dim
        dit_config.intermediate_size = args.dit_config.ffn_dim
        dit_config.num_attention_heads = args.dit_config.num_heads
        dit_config.num_hidden_layers = args.dit_config.num_layers
        dit_config.vocab_size = 0
        
        estimator = VoxCPMLocDiT(dit_config, in_channels=args.feat_dim)
        self.feat_decoder = UnifiedCFM(
            in_channels=args.feat_dim,
            cfm_params=args.dit_config.cfm_config,
            estimator=estimator
        )
        
        # Projections
        self.fsq_layer = ScalarQuantizationLayer(
            args.lm_config.hidden_size,
            args.lm_config.hidden_size,
            args.scalar_quantization_latent_dim,
            args.scalar_quantization_scale
        )
        
        self.enc_to_lm_proj = nn.Linear(args.encoder_config.hidden_dim, args.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(args.lm_config.hidden_size, args.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(args.lm_config.hidden_size, args.dit_config.hidden_dim)
        
        # Stop Predictor
        self.stop_proj = nn.Linear(args.lm_config.hidden_size, args.lm_config.hidden_size)
        self.stop_head = nn.Linear(args.lm_config.hidden_size, 2, bias=False)
        
        # Audio VAE
        self.audio_vae = AudioVAE(args.audio_vae_config)
        
        # Compute chunk_size from encoder rates (hop length)
        self._chunk_size = 1
        for rate in args.audio_vae_config.encoder_rates:
            self._chunk_size *= rate
        
        # Placeholder for tokenizer
        self.tokenizer = None

    @property
    def sample_rate(self) -> int:
        """Return the audio sample rate from the VAE config."""
        return self.args.audio_vae_config.sample_rate

    @property
    def chunk_size(self) -> int:
        """Return the chunk size (hop length) from encoder rates."""
        return self._chunk_size

    @property
    def latent_dim(self) -> int:
        """Return the latent dimension from the VAE config."""
        return self.args.audio_vae_config.latent_dim

    def sanitize(self, weights: dict):
        from mlx.utils import tree_flatten
        
        # Track whether VAE weights were already sanitized to prevent double processing
        vae_already_sanitized = False
        
        # 0. Check if audio_vae weights are present. If not, try to load from pth
        has_vae = any(k.startswith("audio_vae.") for k in weights.keys())
        if not has_vae and self.args.model_path:
            p = Path(self.args.model_path) / "audiovae.pth"
            if p.exists():
                try:
                    import torch
                    state = torch.load(p, map_location="cpu")
                    if "state_dict" in state:
                        state = state["state_dict"]
                    
                    # Convert to numpy/mlx and prefix
                    vae_weights_pth = {}
                    for k, v in state.items():
                        if k.startswith("module."):
                            k = k[7:]
                        # v is tensor
                        arr = mx.array(v.numpy())
                        vae_weights_pth[k] = arr
                        
                    # Sanitize these VAE weights
                    sanitized_vae = self.audio_vae.sanitize(vae_weights_pth)
                    
                    # Add to main weights
                    for k, v in sanitized_vae.items():
                         weights[f"audio_vae.{k}"] = v
                    
                    # Mark that VAE weights have been sanitized
                    vae_already_sanitized = True
                         
                except ImportError:
                    print(f"Warning: torch not installed, skipping loading {p}")
                except Exception as e:
                    print(f"Error loading {p}: {e}")

        # Delegate AudioVAE sanitize (if keys exist now and not already sanitized)
        # Extract audio_vae weights
        vae_weights = {k: v for k, v in weights.items() if k.startswith("audio_vae.")}
        # Strip prefix
        vae_weights_stripped = {k[len("audio_vae."):]: v for k, v in vae_weights.items()}
        
        if vae_weights_stripped and not vae_already_sanitized:
             # Sanitize VAE
             sanitized_vae = self.audio_vae.sanitize(vae_weights_stripped)
             # Put back
             for k in list(vae_weights.keys()):
                 del weights[k]
             for k, v in sanitized_vae.items():
                 weights[f"audio_vae.{k}"] = v
        
        new_weights = {}
        curr_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}
        
        for k, v in weights.items():
            if k not in curr_shapes:
                # might be skipped params or structure mismatch
                # keep it for now
                new_weights[k] = v
                continue
                
            target_shape = curr_shapes[k]
            if v.shape == target_shape:
                new_weights[k] = v
            else:
                # Try transpose
                if len(v.shape) == 2 and v.transpose().shape == target_shape:
                     new_weights[k] = v.transpose()
                else:
                    # Shape mismatch that transpose can't fix
                    # Check for 1D weights (bias, RMSNorm)
                    if len(v.shape) == 1 and len(target_shape) == 1 and v.shape != target_shape:
                         # e.g. embedding size mismatch
                         print(f"Shape mismatch for {k}: {v.shape} vs {target_shape}")
                    new_weights[k] = v
        
        # 3. Add computed buffers (RoPE) if missing
        # Since we are strict loading, we need to provide all params.
        # RoPE params are computed in __init__, so we can grab them from self.
        
        model_params = dict(tree_flatten(self.parameters()))
        for k, v in model_params.items():
            if k not in new_weights and ("rope" in k):
                 # It's a missing RoPE parameter that we have in memory.
                 # Add it.
                 new_weights[k] = v
                    
        return new_weights

    @classmethod
    def post_load_hook(cls, model: 'Model', model_path: Path):
        """
        Hook called after model weights are loaded.
        Used to initialize the tokenizer which is required for text input.
        """
        from transformers import AutoTokenizer
        if model.tokenizer is None:
            model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        return model

    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_audio: mx.array,
    ) -> Dict[str, Any]:
        """
        Build prompt cache for voice cloning.
        
        Args:
            prompt_text: Text spoken in the reference audio
            prompt_audio: Reference audio as MLX array (1D or 2D with shape (samples,) or (1, samples))
            
        Returns:
            dict with prompt_text and audio features for voice cloning
        """
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio[None, :]  # (1, samples)
        
        # If stereo, convert to mono
        if prompt_audio.shape[0] > 1:
            prompt_audio = mx.mean(prompt_audio, axis=0, keepdims=True)
        
        patch_len = self.patch_size * self.chunk_size
        
        # Pad audio to be divisible by patch_len (left padding to preserve end)
        audio_len = prompt_audio.shape[1]
        if audio_len % patch_len != 0:
            padding_size = patch_len - (audio_len % patch_len)
            # Left pad
            prompt_audio = mx.pad(prompt_audio, ((0, 0), (padding_size, 0)))
        
        # Encode audio to get features
        # audio_vae.encode expects (N, T, C) or (N, T) - we have (1, T)
        audio_feat = self.audio_vae.encode(prompt_audio)  # (1, T', latent_dim)
        
        # Reshape to (T_patches, patch_size, latent_dim)
        # audio_feat is (1, T', D) where T' = audio_len / chunk_size
        # We need to reshape to (num_patches, patch_size, D)
        audio_feat = audio_feat.squeeze(0)  # (T', D)
        num_frames = audio_feat.shape[0]
        num_patches = num_frames // self.patch_size
        
        audio_feat = audio_feat[:num_patches * self.patch_size, :]  # Trim to exact multiple
        audio_feat = audio_feat.reshape(num_patches, self.patch_size, self.latent_dim)  # (T, P, D)
        
        return {
            "prompt_text": prompt_text,
            "audio_feat": audio_feat,
        }

    @classmethod
    def from_pretrained(cls, model_path: str):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import numpy as np
        
        model_path = Path(model_path)
        if not model_path.exists():
            model_path = Path(snapshot_download(str(model_path)))
            
        with open(model_path / "config.json") as f:
            config = json.load(f)
            
        args = ModelArgs.from_dict(config)
        model = cls(args)
        
        # Load main weights
        weights = {}
        if (model_path / "model.safetensors").exists():
            with safe_open(model_path / "model.safetensors", framework="numpy") as f:
                for k in f.keys():
                    weights[k] = mx.array(f.get_tensor(k))
        
        # Load AudioVAE weights
        # PyTorch checkpoint for VAE
        if (model_path / "audiovae.pth").exists():
            import torch
            vae_pt = torch.load(model_path / "audiovae.pth", map_location="cpu")
            if "state_dict" in vae_pt:
                vae_pt = vae_pt["state_dict"]
            
            for k, v in vae_pt.items():
                weights[f"audio_vae.{k}"] = mx.array(v.numpy())
                
        # Sanitize
        weights = model.sanitize(weights)
        
        # Load weights
        model.load_weights(list(weights.items()), strict=False)
        
        # Tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model

    def generate(
        self, 
        text: str, 
        ref_audio: mx.array = None, 
        ref_text: str = None,
        inference_timesteps: int = 10, 
        cfg_value: float = 2.0, 
        max_len: int = 4096, 
        **kwargs
    ):
        """
        Generate audio from text, optionally cloning a voice from reference audio.
        
        Args:
            text: Text to synthesize
            ref_audio: Reference audio for voice cloning (MLX array)
            ref_text: Text spoken in the reference audio (required if ref_audio is provided)
            inference_timesteps: Number of diffusion steps (higher = better quality, slower)
            cfg_value: Classifier-free guidance value
            max_len: Maximum generation length in patches
            
        Yields:
            GenerationResult with audio and metadata
        """
        # Tokenize
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
            
        start_time = time.perf_counter()
        
        # Handle voice cloning prompt
        prompt_cache = None
        if ref_audio is not None:
            if ref_text is None:
                raise ValueError("ref_text is required when ref_audio is provided")
            prompt_cache = self.build_prompt_cache(ref_text, ref_audio)
        
        # Build text tokens
        if prompt_cache is not None:
            # Combine prompt text + target text
            full_text = prompt_cache["prompt_text"] + text
            prompt_audio_feat = prompt_cache["audio_feat"]  # (T_prompt, P, D)
        else:
            full_text = text
            prompt_audio_feat = None
            
        input_ids = self.tokenizer.encode(full_text)
        input_ids = mx.array(input_ids)
        token_count = len(input_ids)
        
        # Add audio start token
        start_token = mx.array([101])
        input_ids = mx.concatenate([input_ids, start_token])
        
        # Get text embedding
        scale_emb = self.args.lm_config.scale_emb if not self.args.lm_config.use_mup else 1.0
        text_embed = self.base_lm.embed_tokens(input_ids[None, :])  # (1, L+1, D)
        
        # Prepare for generation with prompt
        if prompt_audio_feat is not None:
            # We have prompt audio features to prepend
            audio_length = prompt_audio_feat.shape[0]
            text_length = text_embed.shape[1]
            
            # Encode prompt audio features
            prompt_feat_4d = prompt_audio_feat[None, :, :, :]  # (1, T_prompt, P, D)
            prompt_feat_embed = self.feat_encoder(prompt_feat_4d)  # (1, T_prompt, H_enc)
            prompt_feat_embed = self.enc_to_lm_proj(prompt_feat_embed)  # (1, T_prompt, H_lm)
            
            # Create masks for the full sequence (text + audio)
            total_length = text_length + audio_length
            text_mask = mx.concatenate([
                mx.ones((1, text_length)),
                mx.zeros((1, audio_length))
            ], axis=1)  # (1, total_length)
            
            audio_mask = mx.concatenate([
                mx.zeros((1, text_length)),
                mx.ones((1, audio_length))
            ], axis=1)  # (1, total_length)
            
            # Combined embeddings: text first, then audio prompt features
            combined_embed = mx.concatenate([text_embed, prompt_feat_embed], axis=1)  # (1, total_length, H)
            
            # Initial prefix condition from last prompt audio frame
            prefix_feat_cond = prompt_audio_feat[-1:, :, :]  # (1, P, D)
            
            print(f"   [Voice cloning] Prompt: {audio_length} audio patches + {text_length} text tokens = {total_length} total")
        else:
            combined_embed = text_embed
            text_mask = mx.ones((1, text_embed.shape[1]))
            audio_mask = mx.zeros((1, text_embed.shape[1]))
            prefix_feat_cond = mx.zeros((1, self.patch_size, self.feat_dim))
        
        # Initial run of base_lm
        print("   Encoding prompt...", end=" ", flush=True)
        enc_outputs, lm_cache = self.base_lm(combined_embed)
        print("done.")
        
        # Apply FSQ only to audio-masked positions (or all for simplicity)
        if prompt_audio_feat is not None:
            # FSQ for audio positions, keep text positions unchanged
            fsq_outputs = self.fsq_layer(enc_outputs)
            enc_outputs = audio_mask[:, :, None] * fsq_outputs + text_mask[:, :, None] * enc_outputs
        
        lm_hidden = enc_outputs[:, -1, :]
        lm_hidden = self.fsq_layer(lm_hidden)
        
        # Residual LM initial run
        print("   Residual encoding...", end=" ", flush=True)
        if prompt_audio_feat is not None:
            # Add feat embed for audio positions
            feat_embed_full = mx.zeros_like(enc_outputs)
            feat_embed_full = mx.concatenate([
                mx.zeros((1, text_embed.shape[1], enc_outputs.shape[-1])),
                prompt_feat_embed
            ], axis=1)
            res_input = enc_outputs + audio_mask[:, :, None] * feat_embed_full
        else:
            res_input = enc_outputs
            
        residual_outputs, res_cache = self.residual_lm(res_input)
        residual_hidden = residual_outputs[:, -1, :]
        print("done.")
        
        # Estimate reasonable max length based on text
        # 1 patch = 0.16 seconds of audio at 44.1kHz
        # Japanese/Chinese: ~4-5 chars per second → ~2 patches per char
        # English: ~3 words per second → ~5 patches per word
        text_chars = len(text)
        estimated_max = min(max(text_chars * 3, 30), max_len)  # More conservative estimate
        
        # Generation Loop
        print(f"   Generating audio patches (max ~{estimated_max}):", end=" ", flush=True)
        pred_feat_seq = []
        
        for i in range(estimated_max):
            if i > 0 and i % 20 == 0:
                print(f"{i}", end=" ", flush=True)
            # DiT
            dit_h1 = self.lm_to_dit_proj(lm_hidden)
            dit_h2 = self.res_to_dit_proj(residual_hidden)
            dit_h = dit_h1 + dit_h2 # (1, H)
            
            cond_in = prefix_feat_cond.transpose(0, 2, 1) # (B, D, P)
            
            pred_feat = self.feat_decoder.sample(
                mu=dit_h,
                n_timesteps=inference_timesteps,
                patch_size=self.patch_size,
                cond=cond_in,
                cfg_value=cfg_value
            )
            
            pred_feat = pred_feat.transpose(0, 2, 1)
            pred_feat_seq.append(pred_feat)
            
            curr_embed = self.feat_encoder(pred_feat[:, None, :, :]) # (B, 1, H)
            curr_embed = self.enc_to_lm_proj(curr_embed)
                
            curr_embed_step = curr_embed # (B, 1, H)
            
            new_lm_out, lm_cache = self.base_lm(inputs_embeds=curr_embed_step, cache=lm_cache)
            
            lm_hidden_next = new_lm_out[:, -1, :]
            lm_hidden_next = self.fsq_layer(lm_hidden_next)
            
            res_in = lm_hidden_next[:, None, :] + curr_embed_step
            new_res_out, res_cache = self.residual_lm(inputs_embeds=res_in, cache=res_cache)
            residual_hidden = new_res_out[:, -1, :]
            
            # Check stop condition AFTER updating hidden state
            stop_logits = self.stop_head(nn.silu(self.stop_proj(lm_hidden_next)))
            stop_flag = mx.argmax(stop_logits, axis=-1).item()
            if i > 5 and stop_flag == 1: 
                break
            
            lm_hidden = lm_hidden_next
            prefix_feat_cond = pred_feat
        
        print(f"done ({len(pred_feat_seq)} patches)")
        print("   Decoding audio...", end=" ", flush=True)
        all_feats = mx.concatenate(pred_feat_seq, axis=1) # (B, Total_P, D)
        
        audio = self.audio_vae.decode(all_feats)
        audio = audio.flatten()
        print("done.")
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        samples = audio.shape[0]
        sample_rate = self.args.audio_vae_config.sample_rate  # Use config value (44100)
        audio_duration_seconds = samples / sample_rate
        
        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0
        
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        # Yield single result
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )
