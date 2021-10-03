import torch


label=torch.Tensor([1,2,3,4])
key=torch.Tensor([0,0,1,0])
sorted, indices = torch.sort(key)
margin = torch.normal(mean=0.5, std=0.05, size=label.size())
margin,_=torch.sort(margin)
print(margin)
print(indices)
margin=margin[indices]
print(margin)