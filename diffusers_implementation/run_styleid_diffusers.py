import torch
import numpy as np, copy, os, sys
import matplotlib.pyplot as plt
import glob
from utils import * # image save utils
from stable_diffusion import load_stable_diffusion, encode_latent, decode_latent, get_text_embedding, get_unet_layers, attention_op  # load SD
import copy
import argparse
import cv2
from tqdm import tqdm

from config import get_args
# Define the class with added methods for resetting state
class style_transfer_module():
           
    def __init__(self,
        unet, vae, text_encoder, tokenizer, scheduler, cfg, style_transfer_params = None,
    ):  
        style_transfer_params_default = {
            'gamma': 0.75,
            'tau': 1.5,
            'injection_layers': [7, 8, 9, 10, 11]
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet 
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.cfg = cfg

        self.attn_features = {} 
        self.attn_features_modify = {} 

        self.cur_t = None
        
        resnet, attn = get_unet_layers(unet)
        self.injection_layers_ids = self.style_transfer_params['injection_layers']
        
        # Initialize dictionary keys for injection layers
        for i in self.injection_layers_ids:
            layer_name = "layer{}_attn".format(i)
            self.attn_features[layer_name] = {}
            # Register hooks
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value(layer_name))
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv(layer_name))
        
        self.trigger_get_qkv = False 
        self.trigger_modify_qkv = False 
        
    def clean_features(self):
        """Clears the attention features to prevent memory leaks and mixing data between runs."""
        for layer_name in self.attn_features:
            self.attn_features[layer_name] = {}
        self.attn_features_modify = {}
        
    def get_text_condition(self, text):
        if text is None:
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.text_encoder.device))[0].to(self.text_encoder.device)
            return {'encoder_hidden_states': uncond_embeddings}
        
        text_embeddings, uncond_embeddings = get_text_embedding(text, self.text_encoder, self.tokenizer)
        text_cond = [text_embeddings, uncond_embeddings]
        denoise_kwargs = {
            'encoder_hidden_states': torch.cat(text_cond)
        }
        return denoise_kwargs
    
    def reverse_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        decode_kwargs = {'vae': self.vae}
        
        for t in self.scheduler.timesteps:
            self.cur_t = t.item()
            with torch.no_grad():
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    input_model = torch.cat([input] * bs)
                else:
                    input_model = input
                
                # Forward pass
                noisy_residual = self.unet(input_model, t.to(input.device), **denoise_kwargs).sample
                    
                if noisy_residual.shape[0] == 2:
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Scheduler step
                # Note: Depending on diffusers version, step() returns a wrapper. 
                # Assuming the provided stable_diffusion.py logic holds:
                step_output = self.scheduler.step(noisy_residual, t, input)
                input = step_output.prev_sample
                
                # Optional: Decode periodically if needed (omitted for speed)
                # pred_latents.append(step_output.pred_original_sample)
                
        # Only decode the final image to save time
        final_image = decode_latent(input, **decode_kwargs)
        return [final_image], [input]

    def invert_process(self, input, denoise_kwargs):
        # ... (Identical to your original logic, keeping it compact here) ...
        # Ensure we return latents list
        pred_latents = []
        decode_kwargs = {'vae': self.vae}

        timesteps = reversed(self.scheduler.timesteps)
        num_inference_steps = len(self.scheduler.timesteps)
        cur_latent = input.clone()

        with torch.no_grad():
            for i in range(0, num_inference_steps):
                t = timesteps[i]
                self.cur_t = t.item()
                
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    cur_latent_input = torch.cat([cur_latent] * bs)
                else:
                    cur_latent_input = cur_latent

                noise_pred = self.unet(cur_latent_input, t.to(cur_latent.device), **denoise_kwargs).sample

                if noise_pred.shape[0] == 2:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

                current_t = max(0, t.item() - (1000//num_inference_steps))
                next_t = t
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]

                if self.cfg.sd_version == "2.1":
                    beta_t = 1 - alpha_t
                    pred_original_sample = alpha_t.sqrt() * cur_latent - beta_t.sqrt() * noise_pred
                    pred_epsilon = alpha_t.sqrt() * noise_pred + beta_t.sqrt() * cur_latent
                    pred_sample_direction = (1 - alpha_t_next).sqrt() * pred_epsilon
                    cur_latent = alpha_t_next.sqrt() * pred_original_sample + pred_sample_direction
                else:
                    cur_latent = (cur_latent - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
                
                pred_latents.append(cur_latent)
        
        return None, pred_latents # Skip intermediate decoding for speed

    # ============================ hook operations ===============================
    def __get_query_key_value(self, name):
        def hook(model, input, output):
            if self.trigger_get_qkv:
                _, query, key, value, _ = attention_op(model, input[0])
                # Save to CPU to save GPU VRAM if caching many
                self.attn_features[name][int(self.cur_t)] = (query.detach().cpu(), key.detach().cpu(), value.detach().cpu())
        return hook

    def __modify_self_attn_qkv(self, name):
        def hook(model, input, output):
            if self.trigger_modify_qkv:
                _, q_cs, k_cs, v_cs, _ = attention_op(model, input[0])
                
                # Retrieve from modify dict (Should be on GPU)
                if int(self.cur_t) in self.attn_features_modify[name]:
                    q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                    
                    # Ensure tensors are on correct device
                    q_c = q_c.to(q_cs.device)
                    k_s = k_s.to(q_cs.device)
                    v_s = v_s.to(q_cs.device)

                    q_hat_cs = q_c * self.style_transfer_params['gamma'] + q_cs * (1 - self.style_transfer_params['gamma'])
                    k_cs, v_cs = k_s, v_s
                    
                    _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_hat_cs, temperature=self.style_transfer_params['tau'])
                    return modified_output
        return hook

def main():
    # --- Configuration ---
    # You can pass these as arguments or edit here
    # Assume args are passed for base config, but we override paths for batching
    cfg = get_args()
    
    # DIRECTORIES
    content_dir = "data/cnt" # Change this to your folder
    style_dir = "data/sty"     # Change this to your folder
    output_dir = cfg.save_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file lists (sorted to match 00.png, 01.png...)
    content_files = sorted(glob.glob(os.path.join(content_dir, "*.png")) + glob.glob(os.path.join(content_dir, "*.jpg")))
    style_files = sorted(glob.glob(os.path.join(style_dir, "*.png")) + glob.glob(os.path.join(style_dir, "*.jpg")))
    
    print(f"Found {len(content_files)} content images and {len(style_files)} style images.")
    
    # Options
    device = "cuda"
    dtype = torch.float16 # Use float16 for speed/memory
    cfg.guidance_scale = 0.0 # Assign to cfg for class access
    
    # --- Load Model Once ---
    print("Loading Stable Diffusion...")
    vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(sd_version=cfg.sd_version, precision_t=dtype)
    scheduler.set_timesteps(cfg.ddim_steps)
    
    # Init module
    style_transfer_params = {'gamma': cfg.gamma, 'tau': cfg.T, 'injection_layers': cfg.layers}
    unet_wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, cfg, style_transfer_params=style_transfer_params)

    # --- PHASE 1: Pre-calculate Content Features (The "Query") ---
    # We invert all content images once and store the results in RAM
    print("Pre-calculating Content Inversions...")
    content_cache = {} # Map filename -> (latent, features)
    
    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False
    
    for c_fn in tqdm(content_files, desc="Inverting Contents"):
        c_name = os.path.basename(c_fn).split('.')[0]
        content_image = cv2.imread(c_fn)[:, :, ::-1]
        
        # Reset wrapper features
        unet_wrapper.clean_features()
        
        # Prepare
        denoise_kwargs = unet_wrapper.get_text_condition(None)
        content_latent = encode_latent(normalize(content_image).to(device=vae.device, dtype=dtype), vae)
        
        # Invert
        _, latents = unet_wrapper.invert_process(content_latent, denoise_kwargs=denoise_kwargs)
        final_latent = latents[-1].detach().cpu() # Move to CPU
        
        # Save features (Deepcopy to avoid reference issues, move to CPU handled in hook)
        features_snapshot = copy.deepcopy(unet_wrapper.attn_features)
        
        content_cache[c_name] = (final_latent, features_snapshot)

    # --- PHASE 2: Iterate Styles and Generate ---
    # We iterate styles, invert ONCE, then apply to all cached contents
    print("Processing Styles and Generating Pairs...")
    
    for s_fn in tqdm(style_files, desc="Style Loop"):
        s_name = os.path.basename(s_fn).split('.')[0]
        style_image = cv2.imread(s_fn)[:, :, ::-1]
        
        # 2a. Invert Style
        unet_wrapper.clean_features()
        unet_wrapper.trigger_get_qkv = True
        unet_wrapper.trigger_modify_qkv = False
        
        denoise_kwargs = unet_wrapper.get_text_condition(None)
        style_latent = encode_latent(normalize(style_image).to(device=vae.device, dtype=dtype), vae)
        
        _, latents = unet_wrapper.invert_process(style_latent, denoise_kwargs=denoise_kwargs)
        style_latent_inverted = latents[-1] # Keep on GPU for generation
        
        # Capture Style Features (Key, Value)
        style_features = copy.deepcopy(unet_wrapper.attn_features)
        # Note: These are on CPU because of our hook modification, we'll move them to GPU when pairing
        
        # 2b. Inner Loop: Apply this style to ALL contents
        for c_name, (c_latent_cpu, c_features_cpu) in content_cache.items():
            
            # Prepare Generation
            unet_wrapper.clean_features()
            unet_wrapper.trigger_get_qkv = False
            unet_wrapper.trigger_modify_qkv = not cfg.without_attn_injection
            
            # Construct Modify Features
            # We need to mix Content (Query) and Style (Key/Value)
            # Both are currently on CPU in dictionaries
            
            for layer_name in style_features.keys():
                unet_wrapper.attn_features_modify[layer_name] = {}
                for t in scheduler.timesteps:
                    t_item = t.item()
                    
                    # Content Q (from cache)
                    q_c = c_features_cpu[layer_name][t_item][0] 
                    # Style K, V (from current style run)
                    k_s = style_features[layer_name][t_item][1]
                    v_s = style_features[layer_name][t_item][2]
                    
                    # We store them on CPU in the modify dict; the hook will move them to GPU
                    unet_wrapper.attn_features_modify[layer_name][t_item] = (q_c, k_s, v_s)
            
            # Prepare Latent for Generation (AdaIN initialization)
            c_latent_gpu = c_latent_cpu.to(device)
            
            if cfg.without_init_adain:
                latent_cs = c_latent_gpu
            else:
                # Calculate AdaIN statistics
                # c_latent_gpu and style_latent_inverted are on GPU
                latent_cs = (c_latent_gpu - c_latent_gpu.mean(dim=(2, 3), keepdim=True)) / \
                            (c_latent_gpu.std(dim=(2, 3), keepdim=True) + 1e-4) * \
                            style_latent_inverted.std(dim=(2, 3), keepdim=True) + \
                            style_latent_inverted.mean(dim=(2, 3), keepdim=True)
            
            # Generate
            images, _ = unet_wrapper.reverse_process(latent_cs, denoise_kwargs=denoise_kwargs)
            
            # Save
            save_name = f"{s_name}_to_{c_name}.jpg"
            image_final = denormalize(images[-1])[0]
            save_image(image_final, os.path.join(output_dir, save_name))

if __name__ == "__main__":
    main()
