import torch

model_state_dict = torch.load('artifacts/checkpoints/best.pt')

head = model_state_dict

with open("resultado.txt", "w") as f:
    f.write(str(head))
    
print("Resultado salvo em 'resultado.txt'")