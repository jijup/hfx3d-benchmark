# Canonical HFX3D colour palette — single source of truth for all Python scripts.
# Hex values match configs/scalar_fields.xml exactly.
# Label order matches configs/label_map.json: 9=other, 10=column, 11=arch.

CLASS_NAMES = [
    'wall', 'window', 'door', 'balcony', 'vegetation', 'stairs',
    'terrain', 'roof', 'blinds', 'other', 'column', 'arch',
]

HEX = [
    '#2F4BFF',  # 0  wall
    '#FF7A00',  # 1  window
    '#00B8D9',  # 2  door
    '#FFD600',  # 3  balcony
    '#00C853',  # 4  vegetation
    '#FF1744',  # 5  stairs
    '#A05A2C',  # 6  terrain
    '#8E24AA',  # 7  roof
    '#00E5A8',  # 8  blinds
    '#9E9E9E',  # 9  other
    '#E040FB',  # 10 column
    '#9CCC00',  # 11 arch
]

RGB_NORM = [
    [0.184, 0.294, 1.000],  # 0  wall
    [1.000, 0.478, 0.000],  # 1  window
    [0.000, 0.722, 0.851],  # 2  door
    [1.000, 0.839, 0.000],  # 3  balcony
    [0.000, 0.784, 0.325],  # 4  vegetation
    [1.000, 0.090, 0.267],  # 5  stairs
    [0.627, 0.353, 0.173],  # 6  terrain
    [0.557, 0.141, 0.667],  # 7  roof
    [0.000, 0.898, 0.659],  # 8  blinds
    [0.620, 0.620, 0.620],  # 9  other
    [0.878, 0.251, 0.984],  # 10 column
    [0.612, 0.800, 0.000],  # 11 arch
]

BY_NAME     = dict(zip(CLASS_NAMES, HEX))
RGB_BY_NAME = dict(zip(CLASS_NAMES, RGB_NORM))

SPLIT = {
    'train': '#2563EB',
    'val':   '#F59E0B',
    'test':  '#DC2626',
}

CHART_STYLE = 'seaborn-v0_8-whitegrid'

CHART_RC = {
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    '#e5e7eb',
    'axes.labelcolor':   '#0f172a',
    'xtick.color':       '#475569',
    'ytick.color':       '#475569',
    'grid.color':        '#e5e7eb',
    'grid.alpha':        0.5,
    'figure.facecolor':  'white',
}
