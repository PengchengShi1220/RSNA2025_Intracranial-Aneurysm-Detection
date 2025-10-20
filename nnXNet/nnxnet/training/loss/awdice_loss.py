import torch

def generate_aneurysm_heatmap(y_true, aneurysm_classes=(14, 26), patch_size=None, power=2.0):
    """
    Efficiently generates an aneurysm heatmap using polynomial decay.

    Args:
        y_true (torch.Tensor): Ground truth labels with shape (b, 1, x, y[, z]).
        aneurysm_classes (tuple): The inclusive range of class values for the aneurysm.
        patch_size (tuple): The size of the patch (x, y[, z]). If None, it is inferred from y_true.
        power (float): The power for the decay, e.g., 2.0 for squared decay.

    Returns:
        torch.Tensor: The heatmap, with shape (b, x, y[, z]), centered on the aneurysm.
    """
    if patch_size is None:
        # Infer spatial dimensions (X, Y, [Z]) from the input tensor.
        patch_size = y_true.shape[2:]

    batch_size = y_true.shape[0]
    dim = len(patch_size) # Spatial dimensionality (2 for 2D, 3 for 3D)
    device = y_true.device

    # 1. Identify Aneurysm Regions
    aneurysm_mask = torch.logical_and(y_true >= aneurysm_classes[0], y_true <= aneurysm_classes[1]).squeeze(1)

    coords_list = [torch.nonzero(mask, as_tuple=False).to(device) for mask in aneurysm_mask]
    
    aneurysm_pixels_batch_indices = [torch.full((len(coords),), b, dtype=torch.long, device=device) 
                                     for b, coords in enumerate(coords_list) if len(coords) > 0]
    
    if not aneurysm_pixels_batch_indices:
        return torch.zeros((batch_size,) + patch_size, dtype=torch.float32, device=device)
        
    all_coords = torch.cat(coords_list, dim=0)
    all_batch_indices = torch.cat(aneurysm_pixels_batch_indices, dim=0)
    
    total_pixels_per_batch = torch.bincount(all_batch_indices, minlength=batch_size).unsqueeze(1).float()
    
    center_mass = torch.zeros(batch_size, dim, dtype=torch.float32, device=device)
    
    center_mass.scatter_add_(0, all_batch_indices.unsqueeze(1).expand(-1, dim), all_coords.float())
    
    center_mass /= (total_pixels_per_batch + 1e-6)

    # 2. Generate Coordinate Grid
    ranges = [torch.arange(size, device=device) for size in patch_size]
    grids = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grids, dim=-1) # Stack to get shape (X, Y, [Z], dim)
    
    expanded_center = center_mass.view(batch_size, *((1,) * dim), dim)
    
    # 3. Compute Polynomial Heatmap
    distances = torch.norm(grid - expanded_center, dim=-1)
    
    max_dist = torch.tensor(patch_size, dtype=torch.float32, device=device).norm()
    
    normalized_distances = distances / (max_dist + 1e-6)
    
    # Apply polynomial decay: 1 - (normalized_distance)^power
    heatmap = 1.0 - torch.pow(normalized_distances, power)
    
    heatmap = torch.clamp(heatmap, min=0.0)
    
    no_aneurysm_mask = (total_pixels_per_batch.squeeze(1) == 0)
    if no_aneurysm_mask.any():
        heatmap[no_aneurysm_mask] = 0.0

    return heatmap

def get_aneurysm_weights(mask_input, y_true, aneurysm_classes=(14, 26), prob_flag=True, aneurysm_weight=1.0):
    """
    Compute weights enhanced by an aneurysm heatmap.
    
    Args:
        mask_input (torch.Tensor): Input mask (predicted or ground truth).
        y_true (torch.Tensor): Ground truth for aneurysm heatmap.
        dim (int): 2 or 3 for 2D/3D.
        prob_flag (bool): Whether input is probabilistic.
        aneurysm_weight (float): Weight for aneurysm heatmap contribution.
    
    Returns:
        torch.Tensor: Weighted mask with shape (b, x, y[, z]).
    """
    if prob_flag:
        mask_prob = mask_input
        mask = (mask_prob > 0.5).int()
    else:
        mask = mask_input

    # Generate aneurysm heatmap
    heatmap = generate_aneurysm_heatmap(y_true, aneurysm_classes=aneurysm_classes, patch_size=mask.shape[1:])
    
    # Enhance weights with heatmap
    weights = 1 + aneurysm_weight * heatmap
    # print("max weights:", weights.max().item())
    
    if prob_flag:
        return weights * mask_prob
    else:
        return weights * mask

class AneurysmWeightedDiceLoss(torch.nn.Module):
    def __init__(self, aneurysm_classes=(17, 29), aneurysm_weight=1.0, smooth=1e-3):
        super(AneurysmWeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.aneurysm_weight = aneurysm_weight
        self.aneurysm_classes = aneurysm_classes

    def forward(self, y_pred, y_true):
        """
        Forward pass for the Dice loss with aneurysm-weighted regions.
        
        Args:
            y_pred (torch.Tensor): Network output with shape (b, c, x, y[, z]).
            y_true (torch.Tensor): Ground truth labels with shape (b, 1, x, y[, z]).
        
        Returns:
            torch.Tensor: Weighted Dice loss.
        """
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        # Process predictions
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]  # Predicted probability map of foreground
        
        with torch.no_grad():
            y_true_binary = torch.where(y_true > 0, 1, 0).squeeze(1).float()  # Ground truth of foreground

        # Compute weighted masks
        w_pred = get_aneurysm_weights(y_pred_prob, y_true, aneurysm_classes=self.aneurysm_classes, prob_flag=True, aneurysm_weight=self.aneurysm_weight)
        w_true = get_aneurysm_weights(y_true_binary, y_true, aneurysm_classes=self.aneurysm_classes, prob_flag=False, aneurysm_weight=self.aneurysm_weight)

        # Compute Dice loss
        intersection = torch.sum(w_pred * w_true, dim=tuple(range(1, len(w_pred.shape))))
        pred_sum = torch.sum(w_pred, dim=tuple(range(1, len(w_pred.shape))))
        true_sum = torch.sum(w_true, dim=tuple(range(1, len(w_true.shape))))
        
        aw_dice = (2.0 * intersection + self.smooth) / (pred_sum + true_sum + self.smooth)
        aw_dice_loss = -aw_dice.mean()
        
        return aw_dice_loss
