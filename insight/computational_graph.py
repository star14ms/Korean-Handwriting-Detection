import torch
from torchviz import make_dot

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kohwctop.model import KoCtoP


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
