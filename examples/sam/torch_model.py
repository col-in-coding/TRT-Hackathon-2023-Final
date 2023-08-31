import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(1280)

    def forward(self, x):
        return self.layernorm(x)


if __name__ == "__main__":

    torch.manual_seed(0)
    inp = torch.zeros(1, 64, 64, 1280).float()
    torch_model = TorchModel()
    torch_model.eval()
    torch.save(torch_model.state_dict(), "state_dict.ckpt")

    torch_out = torch_model(inp)
    print(torch_out.shape, torch_out.sum())
