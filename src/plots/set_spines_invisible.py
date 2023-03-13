def set_spines_invisible(ax, exclude=None):
    for x in ["top", "right", "left", "bottom"]:
        if exclude is None or x != exclude:
            ax.spines[x].set_visible(False)
