import matplotlib.pyplot as plt
from cycler import cycler

COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Orange
    "tertiary": "#009E73",  # Green
    "accent": "#E69F00",  # Yellow
    "accent2": "#CC79A7",  # Pink
    "accent3": "#56B4E9",  # Light blue
    "accent4": "#F0E442",  # Light yellow
    "accent5": "#000000",  # Black
    "light_gray": "#BBBBBB",
    "dark_gray": "#555555",
}


CLASS_COLORS = [
    # Okabe-Ito palette
    "#0072B2",  # Blue
    "#D55E00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#56B4E9",  # Light blue
    "#E69F00",  # Yellow
    "#F0E442",  # Light yellow
    # IBM colorblind-friendly palette
    "#648FFF",  # Blue 2
    "#785EF0",  # Purple
    "#DC267F",  # Magenta
    "#FE6100",  # Orange 2
    "#FFB000",  # Gold
    # Tol Muted palette (colorblind-friendly)
    "#44AA99",  # Cyan
    "#117733",  # Forest green
    "#999933",  # Olive
    "#DDCC77",  # Sand
    "#CC6677",  # Rose
    "#882255",  # Wine
    "#332288",  # Indigo
]


def get_class_visuals(n_classes: int) -> tuple[list[str], list[str]]:
    colors = []
    line_styles = []

    styles = ["-", "--", "-.", ":"]

    for i in range(n_classes):
        colors.append(CLASS_COLORS[i % len(CLASS_COLORS)])
        if n_classes > len(CLASS_COLORS):
            line_styles.append(styles[(i // len(CLASS_COLORS)) % len(styles)])
        else:
            line_styles.append("-")

    return colors, line_styles


color_cycler = cycler(
    color=[
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["tertiary"],
        COLORS["accent"],
        COLORS["accent2"],
        COLORS["accent3"],
    ]
)


def apply_style():
    plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Helvetica",
                "Verdana",
                "sans-serif",
            ],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Figure settings
            "figure.dpi": 300,
            "figure.figsize": (4.5, 4),
            "figure.constrained_layout.use": True,
            # Axes settings
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": color_cycler,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid settings
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "grid.color": COLORS.get("light_gray", "#CCCCCC"),
            # Legend settings
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.fancybox": True,
            "legend.edgecolor": COLORS.get("light_gray"),
            # Savefig settings
            "savefig.dpi": 600,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )
