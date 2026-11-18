# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import plotly
from plotly.subplots import make_subplots

from mars_currents.grid import Cell

# %%
dr = 0.3
dth = np.deg2rad(15)
dphi = np.deg2rad(20)
r = np.arange(1, 3 + dr, dr)
th = np.arange(0 + dth / 2, np.pi, dth)

phi = np.arange(-np.pi, np.pi, dphi)

r_idx = []
th_idx = []
phi_idx = []

for i in range(len(r)):
    for j in range(len(th)):
        for k in range(len(phi)):

            r_idx.append(i)
            th_idx.append(j)
            phi_idx.append(k)

            if th[j] == -np.pi / 2 or th[j] == np.pi / 2:
                break

cells = [
    Cell(dr, dth, dphi, r_idx[i], th_idx[i], phi_idx[i], r, th, phi)
    for i in range(len(r_idx))
]

# %%
cells_list = [
    cell
    for cell in cells[100:]
    if (
        (cell.phi_c <= np.deg2rad(-60) or cell.phi_c >= np.deg2rad(+60))
        and cell.r_c <= 2
        and cell.r_c >= 1
    )
]
cells_plot = cells  # [x  for sublist in cells_list for x in sublist]

# %%
fig = make_subplots()
[
    c.plot_cell(fig, 50, color_faces=True, plot_neighbours=True) for c in cells_plot
]  # if (c.phi_c>30 and c.phi_c<=270 and c.r_c>2) or (c.r_c<=2)]
graph_lim = [-1.9, 1.9]
fig.update_layout(
    title="grid cells",
    # scene = dict(
    #    xaxis = dict(nticks=4, range=graph_lim),
    #    yaxis = dict(nticks=4, range=graph_lim),
    #    zaxis = dict(nticks=4, range=graph_lim)),# autosize=True,
    width=650,
    height=650,
    showlegend=False,
)

# fig.add_trace(go.Scatter3d(x=x_vert,y=y_vert,z=z_vert,mode='markers'))
# fig.add_trace(go.Scatter3d(x=x_neighb,y=y_neighb,z=z_neighb,mode='markers'))

fig.show()
# plotly.offline.plot(fig, filename='lifeExp.html')

# %%
plotly.offline.plot(fig, filename="cell_neighbours.html")

# %%
