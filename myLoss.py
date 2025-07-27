import torch

class DualTripletMarginLoss(torch.nn.TripletMarginWithDistanceLoss):
    """
    对于三元组 (anchor, positive, negative)
    同时要求ap距离和pn距离，使得类内和类间距离都有约束
    """
    def __init__(self, margin_pos=0.2, margin_neg=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        # self.margin_neg_current = margin_neg * 0.5
        # self.margin_neg_a = margin_neg * 0.5

    def forward(self, anchor, positive, negative):
        d_ap = self.distance_function(anchor, positive)
        d_an = self.distance_function(anchor, negative)
        
        # # 带缓冲的设计，类间的距离是可以短暂妥协的
        # loss_ap_mean = torch.relu(d_ap - self.margin_pos).mean()
        # loss_an_mean = torch.relu(self.margin_neg_current - d_an).mean()
        # loss_an_mean_ = loss_an_mean.detach().item()
        # if loss_an_mean_ > 0:
        #     self.margin_neg_current = min(self.margin_neg_current - loss_an_mean_ + self.margin_neg_a, self.margin_neg)
        #     self.margin_neg_a += loss_an_mean_ / 10
        # else:
        #     self.margin_neg_current = self.margin_neg
        #     self.margin_neg_a = 0

        # return loss_ap_mean + loss_an_mean, self.margin_neg_current
        
        
        # 极简版
        loss = (torch.relu(d_ap - self.margin_pos) + 
                torch.relu(self.margin_neg - d_an))
        return loss.mean(), self.margin_neg

# [Obsolete]
class GeneralizedDynamicLoss(torch.nn.Module):
    '''
    通用的动态损失加权
    实际收敛效果不如fix

    # 示例：支持N个损失的动态加权
    loss_fn = GeneralizedDynamicLoss(base_weights=[0.4, 0.3, 0.3])
    ...
    loss, norm_weights = loss_fn(name_loss, prefix_loss, star_loss)
    print(f"LossNormWeights: {norm_weights.cpu().numpy()}")
    '''
    def __init__(self, base_weights):
        super().__init__()
        self.base_weights = torch.nn.Parameter(torch.tensor(base_weights), requires_grad=False)
        
    def forward(self, *losses):
        assert len(losses) == len(self.base_weights)
        
        losses_tensor = torch.stack(losses)
        sum_losses = losses_tensor.sum() + 1e-8
        dynamic_weights = self.base_weights.to(losses_tensor.device) * (losses_tensor / sum_losses)
        norm_weights = dynamic_weights / (dynamic_weights.sum() + 1e-8)
            
        return (norm_weights * losses_tensor).sum(), norm_weights.detach()