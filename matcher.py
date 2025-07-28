import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

def generalized_box_iou(boxes1, boxes2):
    # Simplified: using normal IoU for now (for speed)
    # In full DETR, implement full GIoU
    return box_iou(boxes1, boxes2)

class HungarianMatcher:
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, Q, C]
        out_bbox = outputs["pred_boxes"]               # [B, Q, 4]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]    # [T]
            tgt_boxes = targets[b]["boxes"]   # [T, 4]

            # ---- Skip if there are no ground truth boxes ----
            if tgt_boxes.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            # classification cost: negative prob of ground truth class
            cost_class = -out_prob[b][:, tgt_ids]

            # L1 bbox cost
            cost_bbox = torch.cdist(out_bbox[b], tgt_boxes, p=1)

            # IoU cost
            cost_giou = 1 - generalized_box_iou(out_bbox[b], tgt_boxes)

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.cpu()

            # Hungarian matching
            i, j = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64)
            ))
        return indices

