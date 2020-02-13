import torch

def print_model_summary(model:torch.nn.Module)->None:
  total_parameters = sum([p.nelement() for p in model.parameters()])
  total_size = sum([p.nelement()*p.element_size() for p in model.parameters()])
  for name, param in model.named_parameters():
    num_params = param.nelement()
    req_grad = "Grad" if param.requires_grad else "No Grad"
    param_percent = num_params / total_parameters * 100
    size_kb = int(param.element_size() * num_params / 1000)
    print(f"""
    {name}
      - {list(param.shape)}
      - {req_grad}
      - {num_params} Params ({param_percent:1.5f})
      - {size_kb} KB
    """.strip())
  print("# Params:", total_parameters)
  print("Size:", total_size*1e-9, "GB")
