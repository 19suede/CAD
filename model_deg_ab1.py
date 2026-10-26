"""
Linear SRF, Content(DWT-based)-based+MoE, Final Version, Tag: ab1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssim import SSIM as ssim
from torch.autograd import Function

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            dx_ll, dx_lh, dx_hl, dx_hh = dx[:, :C], dx[:, C:C * 2], dx[:, C * 2:C * 3], dx[:, C * 3:]

            dx_x_ll = torch.nn.functional.conv_transpose2d(dx_ll, w_ll.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_lh = torch.nn.functional.conv_transpose2d(dx_lh, w_lh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hl = torch.nn.functional.conv_transpose2d(dx_hl, w_hl.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hh = torch.nn.functional.conv_transpose2d(dx_hh, w_hh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            return dx_x_ll + dx_x_lh + dx_x_hl + dx_x_hh, None, None, None, None
        else:
            return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self):
        super(DWT_2D, self).__init__()
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)

        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)

        self.w_ll = w_ll.to(dtype=torch.float32)
        self.w_lh = w_lh.to(dtype=torch.float32)
        self.w_hl = w_hl.to(dtype=torch.float32)
        self.w_hh = w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
    
class MoEKernelExpert(nn.Module):
    def __init__(self, embed_dim, kernel_size, subscale_ratio=3, hidden=128):
        super().__init__()
        K = kernel_size
        self.kernel_size = K
        self.sub_K = K // subscale_ratio
        self.main = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, K*K),
        )

    def forward(self, e):
        B = e.shape[0]
        kernel = self.main(e).view(B, 1, self.kernel_size, self.kernel_size)  
        kernel = F.softmax(kernel.view(B, -1), dim=1).view(B, 1, self.kernel_size, self.kernel_size)

        kernel = kernel / (kernel.sum(dim=[2,3], keepdim=True) + 1e-12)

        return kernel  # (B,1,K,K)

class DynamicBlurNet(nn.Module):
    def __init__(self,
                 embedding_dim,
                 scale=4,
                 kernel_size=21,
                 num_experts=8,
                 top_k=8):
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size

        self.router = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_experts)
        )

        # experts
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.experts = nn.ModuleList([
            MoEKernelExpert(embedding_dim, kernel_size, hidden=embedding_dim)
            for _ in range(num_experts)
        ])

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.downsampler = nn.AvgPool2d(kernel_size=scale)

    def _router_probs_topk(self, emb):
        # emb: (B, embed_dim)
        logits = self.router(emb)  # (B, E)
        if self.top_k >= self.num_experts:
            probs = F.softmax(logits, dim=1)
            return probs, None
        topv, topi = torch.topk(logits, k=self.top_k, dim=1)
        topv = F.softmax(topv, dim=1)
        sparse = torch.zeros_like(logits)
        sparse.scatter_(1, topi, topv)
        return sparse, (topv, topi)

    def compute_balance_loss(self, router_probs):
        mean_prob = router_probs.mean(dim=0)      # (E,)
        E = mean_prob.numel()
        uniform = 1.0 / E
        loss = E * torch.abs(mean_prob - uniform).sum()
        return loss

    def forward(self, x, emb):
        B, C, H, W = x.shape

        router_probs, topk_info = self._router_probs_topk(emb)  # (B, E)

        expert_kernels = []
        for j, expert in enumerate(self.experts):
            kj = expert(emb)  # (B,1,K,K)
            expert_kernels.append(kj)
        # stack: (E, B, 1, K, K) -> transpose to (B, E, 1, K, K)
        expert_kernels = torch.stack(expert_kernels, dim=0).permute(1, 0, 2, 3, 4)

        # router_probs: (B, E) -> (B, E, 1, 1, 1)
        w = router_probs.view(B, self.num_experts, 1, 1, 1)
        kernel = (expert_kernels * w).sum(dim=1)  # (B, 1, K, K)
        kernel = kernel.repeat_interleave(C, dim=0)  # (B*C, 1, K, K)

        # apply depthwise grouped conv
        x_padded = self.pad(x)
        x_padded = x_padded.reshape(1, B*C, x_padded.shape[2], x_padded.shape[3])
        out = F.conv2d(x_padded, kernel, groups=B*C).reshape(B, C, H, W)

        out = out[:, :, 1::self.scale, 1::self.scale]
        self._last_router_probs = router_probs

        return out
    

    def last_router_probs(self, detach=True):
        probs = getattr(self, "_last_router_probs", None)
        if probs is None:
            return None
        return probs.detach() if detach else probs
    
    def get_kernels(self, emb):
        B = emb.shape[0]

        router_probs, _ = self._router_probs_topk(emb)  # (B,E)

        expert_kernels = []
        for expert in self.experts:
            k = expert(emb)  # (B,1,K,K)
            expert_kernels.append(k)

        expert_kernels = torch.stack(expert_kernels, dim=1)  # (B,E,1,K,K)

        w = router_probs.view(B, self.num_experts, 1, 1, 1)
        final_kernel = (expert_kernels * w).sum(dim=1)  # (B,1,K,K)

        expert_kernels = expert_kernels.squeeze(2)  # (B,E,K,K)
        final_kernel = final_kernel.squeeze(1)      # (B,K,K)

        return expert_kernels, final_kernel, router_probs


class GlobalSceneEncoder(nn.Module):
    def __init__(self, ms_channels, embed_dim):
        super().__init__()
        self.ms_channels = ms_channels
        self.embed_dim = embed_dim
        fusion_in = ms_channels + 1
        self.dwt = DWT_2D()

        self.fuse_head = nn.Sequential(
            nn.Linear(embed_dim*4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        def make_head(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(embed_dim, embed_dim)
            )
        self.high_head1 = make_head(1)
        self.high_head2 = make_head(1)
        self.high_head3 = make_head(1)
        self.low_head = make_head(1)
        


    def forward(self, pan, ms):
        pan_wt = self.dwt(pan)
        pan_low = pan_wt[:, :1]
        pan_high1 = pan_wt[:, 1:2]
        pan_high2 = pan_wt[:, 2:3]
        pan_high3 = pan_wt[:, 3:4]
        pan_low = self.low_head(pan_low)
        pan_high1 = self.high_head1(pan_high1)
        pan_high2 = self.high_head2(pan_high2)
        pan_high3 = self.high_head3(pan_high3)
        x = torch.cat([pan_low, pan_high1, pan_high2, pan_high3], dim=1)
        scene_embed = self.fuse_head(x)
        # print(scene_embed.shape)

        return scene_embed
    

    
class RegistrationNet(nn.Module):
    def __init__(self, in_channels=5, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential( 
            nn.Linear(2, 200), 
            nn.ReLU(),
            nn.Linear(200, 50), 
            nn.ReLU(),
            nn.Linear(50, 2), 
        ) 

    def forward(self, pan, ms):
        H, W = pan.shape[-2:]
        device = pan.device
        y, x = torch.meshgrid( 
            torch.linspace(-1,1,H,device=device), 
            torch.linspace(-1,1,W,device=device), indexing='ij' 
            ) 
        coord = torch.stack([x,y], dim=-1) 
        coord = coord.view(-1,2) 
        flow = self.mlp(coord) 
        flow = flow.view(1,H,W,2).permute(0,3,1,2) 
        
        return flow

def warp(x, flow):
    B, C, H, W = x.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1,1,H,device=x.device),
        torch.linspace(-1,1,W,device=x.device),
        indexing='ij'
    )

    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    base_grid = base_grid.unsqueeze(0).repeat(B,1,1,1)

    flow_norm = torch.zeros_like(flow)
    flow_norm[:,0] = flow[:,0] / ((W-1)/2)
    flow_norm[:,1] = flow[:,1] / ((H-1)/2)

    grid = base_grid + flow_norm.permute(0,2,3,1)

    out = F.grid_sample(
        x, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return out

    
class Degradation(nn.Module):
    def __init__(self, num_ms_channels=4, scale=4,
                 embedding_dim=32, blur_kernel_size=15):
        super().__init__()
        self.num_ms_channels = num_ms_channels
        
        self.registration = RegistrationNet(
            in_channels=num_ms_channels + 1
        )

        self.scene_encoder = GlobalSceneEncoder(num_ms_channels, embedding_dim)

        self.DegNet = DynamicBlurNet(
            embedding_dim,
            scale=scale, kernel_size=blur_kernel_size,
        )

        self.srf = nn.Sequential(
            nn.Conv2d(num_ms_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.ssim_loss = ssim()
        self.l1_loss = nn.L1Loss()


    def forward(self, pan, ms):
        balance_list = []

        flow = self.registration(pan, ms)
        pan = warp(pan, flow)

        scene_emb = self.scene_encoder(pan, ms)
        
        d_pan = self.DegNet(pan, scene_emb)
        balance_list.append(self.DegNet.last_router_probs(detach=False))
        d_pan = torch.clamp(d_pan, 0, 1)

        srf_weight = F.softmax(self.srf[0].weight, dim=1)
        pan_lf_target = F.conv2d(ms, srf_weight)
        pan_lf_target = torch.clamp(pan_lf_target, 0, 1)
        
        l_srf_lvl1 = 1 - self.ssim_loss(d_pan, pan_lf_target)
        balance = self.DegNet.compute_balance_loss(torch.cat(balance_list, dim=0))
        

        total_loss = (
            l_srf_lvl1 + 
            1e-3 * balance 
        )

        return total_loss, l_srf_lvl1, l_srf_lvl1, balance

    def infer(self, pan, ms):
        self.eval()

        with torch.no_grad():
            flow = self.registration(pan, ms)
            pan = warp(pan, flow)

            scene_emb = self.scene_encoder(pan, ms)
            d_pan = self.DegNet(pan, scene_emb)
            d_ms = self.DegNet(ms, scene_emb)

        d_pan = torch.clamp(d_pan, 0, 1)
        d_ms = torch.clamp(d_ms, 0, 1)
        return d_pan, d_ms
    
    def infer_pan(self, pan, ms):
        self.eval()

        with torch.no_grad():
            flow = self.registration(pan, ms)
            pan = warp(pan, flow)

            scene_emb = self.scene_encoder(pan, ms)

        return scene_emb

    # inference at full-resolution test stage, before downstream fusion networks
    def reg(self, pan, ms):
        self.eval()
        with torch.no_grad():
            flow = self.registration(pan, ms)
            pan_aligned = warp(pan, flow)
            
        return pan_aligned, flow
    
    