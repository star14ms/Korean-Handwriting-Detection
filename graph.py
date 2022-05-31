import torch
from torchviz import make_dot

from model import KoCtoP


device = "cuda" if torch.cuda.is_available() else "cpu"
model = KoCtoP().to(device)
model.eval()

kwargs = {
    'fontname': 'BM JUA_TTF',
    'fontsize': '16',
}

x = torch.randn([10, 1, 64, 64], device=device)
yi, ym, yf = model(x)
make_dot((yi, ym, yf), params=dict(model.named_parameters()), **kwargs).render("insight/graph", format="png")
