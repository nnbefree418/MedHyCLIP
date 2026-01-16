import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from PIL import Image
from CLIP.tokenizer import tokenize
import geoopt  # 双曲几何库（用于 Hyper-MVFA）


def encode_text_with_prompt_ensemble(model, obj, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def encode_text_with_hyperbolic_adjustment(model, obj, device, 
                                           use_hyperbolic=False, 
                                           c=0.1, 
                                           scale_normal=0.1, 
                                           scale_abnormal=0.8):
    """
    Encode text with optional hyperbolic space projection and radius adjustment.
    
    This function wraps the original encode_text_with_prompt_ensemble and adds
    hyperbolic geometric transformations when use_hyperbolic=True.
    
    Key innovation: Radius adjustment forces 'normal' prompts toward the ball center
    (small radius, general concept) and 'abnormal' prompts toward the edge
    (large radius, specific anomaly).
    
    Args:
        model: CLIP model for text encoding
        obj: Object/dataset name (e.g., 'Liver', 'Brain')
        device: Torch device (cuda/cpu)
        use_hyperbolic: Whether to use hyperbolic mode (default: False)
        c: Curvature of Poincaré ball (default: 0.1)
        scale_normal: Radius scaling for normal prompts (default: 0.1, toward center)
        scale_abnormal: Radius scaling for abnormal prompts (default: 0.8, toward edge)
    
    Returns:
        If use_hyperbolic=False:
            (text_features, None) - Euclidean features [C, 2], no ball object
        If use_hyperbolic=True:
            (text_features_hyp, ball) - Hyperbolic features [C, 2], ball object
    """
    # Step 1: Get Euclidean text features using original ensemble method
    text_features = encode_text_with_prompt_ensemble(model, obj, device)  # [C, 2]
    
    if not use_hyperbolic:
        # Return Euclidean features directly (original MVFA behavior)
        return text_features, None
    
    # Step 2: Hyperbolic mode - project to Poincaré ball and adjust radii
    ball = geoopt.PoincareBall(c=c)
    
    # Transpose to [2, C] for easier manipulation (index 0=normal, 1=abnormal)
    text_e = text_features.T  # [2, C]
    
    # Normalize to unit sphere before projection (improves stability)
    text_e = text_e / text_e.norm(dim=-1, keepdim=True)
    
    # Step 3: Project to hyperbolic space via exponential map
    text_h = ball.expmap0(text_e)  # [2, C]
    
    # Step 4: Separate normal and abnormal features
    normal_feat = text_h[0]    # [C]
    abnormal_feat = text_h[1]  # [C]
    
    # Step 5: Radius adjustment using Möbius scalar multiplication
    # - scale_normal (0.1): shrink normal toward center (root of hierarchy)
    # - scale_abnormal (0.8): push abnormal toward edge (leaf/specific anomaly)
    # Convert scalars to tensors for geoopt compatibility
    scale_normal_tensor = torch.tensor(scale_normal, dtype=normal_feat.dtype, device=normal_feat.device)
    scale_abnormal_tensor = torch.tensor(scale_abnormal, dtype=abnormal_feat.dtype, device=abnormal_feat.device)
    normal_hyp = ball.mobius_scalar_mul(scale_normal_tensor, normal_feat)
    abnormal_hyp = ball.mobius_scalar_mul(scale_abnormal_tensor, abnormal_feat)
    
    # Step 6: Stack and transpose back to [C, 2] format
    text_final = torch.stack([normal_hyp, abnormal_hyp], dim=0)  # [2, C]
    text_final = text_final.T  # [C, 2] - maintains interface compatibility
    
    return text_final, ball


def hyperbolic_distance_batch(memory_features, test_features, ball, chunk_size=256, mem_chunk_size=100):
    """
    Compute pairwise hyperbolic distances with memory-efficient chunking.
    
    This function splits the computation into smaller batches to avoid OOM errors.
    
    Args:
        memory_features: [N_mem, C] - Hyperbolic features from memory bank
        test_features: [L, C] - Hyperbolic features from test patches
        ball: geoopt.PoincareBall object
        chunk_size: Maximum number of test patches to process at once (default: 256)
        mem_chunk_size: Maximum number of memory features to process at once (default: 100)
    
    Returns:
        Distance matrix [N_mem, L]
        dist[i, j] = hyperbolic_distance(memory_features[i], test_features[j])
    
    Implementation:
        Splits both memory_features and test_features into chunks to avoid OOM.
    """
    N_mem, C = memory_features.shape
    L, _ = test_features.shape
    
    # Initialize result tensor
    all_distances = []
    
    # Process memory features in chunks
    for mem_start in range(0, N_mem, mem_chunk_size):
        mem_end = min(mem_start + mem_chunk_size, N_mem)
        mem_chunk = memory_features[mem_start:mem_end]  # [chunk_N_mem, C]
        chunk_N_mem = mem_chunk.shape[0]
        
        distances_for_mem_chunk = []
        
        # Process test features in chunks
        for test_start in range(0, L, chunk_size):
            test_end = min(test_start + chunk_size, L)
            test_chunk = test_features[test_start:test_end]  # [chunk_L, C]
            chunk_L = test_chunk.shape[0]
            
            # Expand dimensions for broadcasting
            # mem_chunk: [chunk_N_mem, C] -> [chunk_N_mem, 1, C] -> [chunk_N_mem, chunk_L, C]
            mem_exp = mem_chunk.unsqueeze(1).expand(chunk_N_mem, chunk_L, C)
            
            # test_chunk: [chunk_L, C] -> [1, chunk_L, C] -> [chunk_N_mem, chunk_L, C]
            test_exp = test_chunk.unsqueeze(0).expand(chunk_N_mem, chunk_L, C)
            
            # Flatten to [chunk_N_mem*chunk_L, C] for batch distance computation
            mem_flat = mem_exp.reshape(chunk_N_mem * chunk_L, C)
            test_flat = test_exp.reshape(chunk_N_mem * chunk_L, C)
            
            # Compute distances for this chunk
            dist_chunk = ball.dist(mem_flat, test_flat)  # [chunk_N_mem*chunk_L]
            
            # Reshape to [chunk_N_mem, chunk_L] and store
            dist_chunk = dist_chunk.reshape(chunk_N_mem, chunk_L)
            distances_for_mem_chunk.append(dist_chunk)
        
        # Concatenate all test chunks along L dimension
        mem_distances = torch.cat(distances_for_mem_chunk, dim=1)  # [chunk_N_mem, L]
        all_distances.append(mem_distances)
    
    # Concatenate all memory chunks along N_mem dimension
    distances = torch.cat(all_distances, dim=0)  # [N_mem, L]
    
    return distances


def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x

def vflip_img(x):
    x = K.geometry.transform.vflip(x)
    return x

def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x


def augment(fewshot_img, fewshot_mask=None):

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

            rotate_mask = rot_img(fewshot_mask, angle)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)
        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

            trans_mask = translation_img(fewshot_mask, a, b)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        flipped_mask = hflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_mask = vflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:
        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])
    
    return augment_fewshot_img, augment_fewshot_mask

