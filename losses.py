import torch
import torch.nn as nn
import torch.nn.functional as F

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        """
        outputs: dict with:
            "pred_logits": [B, Q, num_classes]
            "pred_boxes":  [B, Q, 4]
        targets: list of dicts with keys ["labels", "boxes"]
        """
        # Perform Hungarian matching
        indices = self.matcher(outputs, targets)
        return self.compute_loss(outputs, targets, indices)

    def compute_loss(self, outputs, targets, indices):
        # --- Classification loss ---
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"][idx]
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        loss_ce = F.cross_entropy(src_logits, target_classes)

        # --- Bounding box loss ---
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes)

        # Combine
        loss = (
            self.weight_dict.get("loss_ce", 1.0) * loss_ce
            + self.weight_dict.get("loss_bbox", 1.0) * loss_bbox
        )
        return loss

    def _get_src_permutation_idx(self, indices):
        # flatten index pairs
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
