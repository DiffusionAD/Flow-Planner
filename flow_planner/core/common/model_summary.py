import torch
from torchinfo import summary

def model_summary(model, data, sample_steps, device, preprocess):
        model_inputs, all_gt, _ = preprocess(data)
        
        B, P, T, D = all_gt.shape 
        z = torch.randn_like(all_gt[...,:4], device=device).view(B, P, -1)
        t = torch.rand(all_gt.shape[0], device=device)
        dt = torch.ones(all_gt.shape[0], device=device) * torch.log2(torch.tensor([sample_steps], device=device).to(torch.int32))
        model_inputs.update({'dt': dt})
        
        summary(model, input_data={"x": z, "t": t, "inputs": model_inputs})