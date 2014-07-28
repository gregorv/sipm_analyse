

def info_box(ax, txt, x=0.75, y=0.95):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=14,
            horizontalalignment="left", verticalalignment='top',
            bbox=props)

