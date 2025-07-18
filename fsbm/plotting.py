

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Ellipse, Rectangle
from .opinion import est_directional_similarity, proj_pca
from .obstacles import obstacle_cfg_vneck, obstacle_cfg_stunnel
import torch


cmap = "Greens"
fontsize = 10

plt.rcParams.update({"font.size": fontsize})

cpuize = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t

################################################################################################


def get_fig_axes(ncol, nrow=1, ax_length_in=2.0, lim=None):
    figsize = (ncol * ax_length_in, nrow * ax_length_in)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, ncol)

    if lim is not None:
        axs = [axes] if nrow == 1 and ncol == 1 else axes.reshape(-1)
        for ax in axs:
            ax.set(xlim=[-lim, lim], ylim=[-lim, lim])

    return fig, axes


def save_fig(fn, pdf=False):
    plt.tight_layout()
    if pdf:
        plt.savefig(f"figs/{fn}.pdf")
    else:
        plt.savefig(f"figs/{fn}.png", dpi=300)
    plt.close()


def get_colors(n_snapshot, cmap=cmap):
    cm1 = cm.get_cmap(cmap)
    colors = cm1(np.linspace(0.2, 0.8, n_snapshot))
    return colors


@torch.no_grad()
def plot_scatter(ax, x, s=2, c=None, zorder=0, marker=None, title=None, alpha=1.0):
    """
    x: (B, 2)
    """
    x = cpuize(x)
    ax.scatter(x[:, 0], x[:, 1], s=s, c=c, marker=marker, zorder=zorder, alpha=alpha)
    if title:
        ax.set_title(title)


@torch.no_grad()
def plot_traj(ax, xs, title=None, **kwargs):
    """
    xs: (B, T, D)
    """
    for x in xs:
        x = cpuize(x)
        ax.plot(x[:, 0], x[:, 1], **kwargs)
    if title:
        ax.set_title(title)


@torch.no_grad()
def plot_boundaries(p0, p1, pcfg):
    fig, axs = get_fig_axes(ncol=2, lim=pcfg.lim)

    plot_scatter(axs[0], p0(512))
    plot_scatter(axs[1], p1(512))

    axs[0].set_title(r"$\mu$ at $t$=0", fontsize=fontsize)
    axs[1].set_title(r"$\nu$ at $t$=1", fontsize=fontsize)

    plot_obstacles(axs[0], pcfg.name)
    plot_obstacles(axs[1], pcfg.name)


def plot_obstacles(ax, name, zorder=0):
    if name == "vneck":
        c_sq, coef = obstacle_cfg_vneck()
        x = np.linspace(-6, 6, 100)
        y1 = np.sqrt(c_sq + coef * np.square(x))
        y2 = np.ones_like(x) * y1[0]

        ax.fill_between(x, y1, y2, color="darkgray", edgecolor=None, zorder=zorder)
        ax.fill_between(x, -y1, -y2, color="darkgray", edgecolor=None, zorder=zorder)

    elif name == "stunnel":
        a, b, cc, centers = obstacle_cfg_stunnel()
        for c in centers:
            elp = Ellipse(
                xy=np.array(c),
                width=2 * np.sqrt(cc / a),
                height=2 * np.sqrt(cc / b),
                zorder=zorder,
            )

            ax.add_artist(elp)
            elp.set_clip_box(ax.bbox)
            elp.set_facecolor("darkgray")
            elp.set_edgecolor(None)



@torch.no_grad()
def plot_directional_sim(ax, xt):
    """
    xt: (B, 2)
    """

    B, D = xt.shape
    assert D == 2

    n_est = 5000
    directional_sim = est_directional_similarity(xt, n_est)
    assert directional_sim.shape == (n_est,)

    directional_sim = directional_sim.detach().cpu().numpy()

    bins = 15
    _, _, patches = ax.hist(
        directional_sim,
        bins=bins,
    )

    colors = plt.cm.coolwarm(np.linspace(1.0, 0.0, bins))

    for c, p in zip(colors, patches):
        plt.setp(p, "facecolor", c)

    ymax = 1000 if xt.shape[1] == 2 else 2000
    ax.relim()
    ax.autoscale()
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)


def show_image(images, ncol=10):

    images = torch.clamp((images + 1) / 2, 0.0, 1.0)

    n = len(images)
    nrow = n // ncol + (1 if n % ncol > 0 else 0)
    assert ncol * nrow >= n

    fig, axs = get_fig_axes(nrow=nrow, ncol=ncol, ax_length_in=1)
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    images = cpuize(images)
    for ax, image in zip(axs.reshape(-1), images):
        ax.imshow(image.permute(1, 2, 0).numpy())
    plt.tight_layout()


@torch.no_grad()
def plot_gpath_nd(result):
    fig, axs = get_fig_axes(ncol=2, ax_length_in=2.5)

    ## Plot gamma (only the first pair)
    axs[0].plot(result["init_gamma"][0], "-x")
    axs[0].plot(result["final_gamma"][0], "-x")
    axs[0].set_title(r"$\gamma(t)$ Init vs Optimized")

    ## Plot loss
    losses = result["losses"]
    axs[1].plot(losses)
    axs[1].set_title(f"Loss, last={losses[-1]:.1f}")


@torch.no_grad()
def plot_gpath_2d(result, pcfg):

    fig, axs = get_fig_axes(ncol=5, ax_length_in=2.5, lim=pcfg.lim)
    for ax in [axs[0], axs[1], axs[4]]:
        plot_obstacles(ax, result["name"])
    for ax in [axs[2], axs[3]]:
        ax.relim()
        ax.autoscale()

    B, T, D = result["init_mean"].shape

    ## Plot mean & std (only the first pair)
    colors = get_colors(T)
    plot_scatter(axs[0], result["init_mean"][0], c=colors, title="Init Mean")
    plot_scatter(axs[1], result["final_mean"][0], c=colors, title="Optimized Mean")
    axs[2].plot(result["init_gamma"][0], "-x")
    axs[2].plot(result["final_gamma"][0], "-x")
    axs[2].set_title(r"$\gamma(t)$ Init vs Optimized")

    ## Plot loss
    losses = result["losses"]
    axs[3].plot(losses)
    axs[3].set_title(f"Loss, last={losses[-1]:.1f}")

    ## Plot marginal xt
    mB, S, N = 512, 5, 64
    with torch.no_grad():
        xt = result["gpath"].sample_xt(torch.linspace(0, 1, S), N=N)  # (B, N, S, D)
        xt = xt.permute(2, 0, 1, 3).reshape(S, B * N, D)
    for i, x in enumerate(xt):
        rand_idx = torch.randperm(B * N)[:mB]
        plot_scatter(axs[4], x[rand_idx], c=f"C{i}")
    axs[4].set_title(f"Optimized Xt")


@torch.no_grad()
def plot_gpath(result, pcfg):
    if result["gpath"].D == 2:
        # crowd nav, opinion 2D
        plot_gpath_2d(result, pcfg)
    else:
        plot_gpath_nd(result)

@torch.no_grad()
def save_xs(t, xs, log_steps, direction, pcfg, fn):
    if pcfg.name == "opinion":
        plot_xs_opinion(t, xs, log_steps, direction, pcfg)
    else:
        plot_xs_crowd_nav(t, xs, log_steps, direction, pcfg)
    save_fig(fn)


@torch.no_grad()
def plot_xs_opinion(t, xs, log_steps, direction, pcfg):
    """
    t: (T,)
    xs: (B, T, D)
    """
    B, T, D = xs.shape
    assert t.shape == (T,) and log_steps <= T

    ### PCA projection
    xs = xs.detach().clone()
    if D > 2:
        xs, _ = proj_pca(xs)
    assert xs.shape == (B, T, 2)

    ### Plot
    fig, axs = get_fig_axes(nrow=2, ncol=log_steps, lim=pcfg.lim)
    steps_to_log = np.linspace(0, T - 1, log_steps).astype(int)
    for i, step in enumerate(steps_to_log):
        plot_scatter(axs[0, i], xs[:, step], s=2, zorder=0)
        axs[0, i].set_title(r"$t$=" + f"{t[step]:.2f}")
        plot_directional_sim(axs[1, i], xs[:, step])

    axs[0, 0].set_ylabel(direction)
    if D > 2:
        axs[1, 0].set_ylabel("PCA 1 vs 2")


@torch.no_grad()
def plot_xs_crowd_nav(t, xs, log_steps, direction, pcfg):
    B, T, D = xs.shape
    assert t.shape == (T,) and log_steps <= T

    fig, axs = get_fig_axes(ncol=log_steps, lim=pcfg.lim)
    steps_to_log = np.linspace(0, T - 1, log_steps).astype(int)

    for ax, step in zip(axs, steps_to_log):
        plot_obstacles(ax, pcfg.name)
        plot_scatter(ax, xs[:512, step])
        ax.set_title(r"$t$=" + f"{t[step]:.2f}")
    axs[0].set_ylabel(direction)
