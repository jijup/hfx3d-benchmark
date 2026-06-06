# Canonical HFX3D colour palette — single source of truth for all Python scripts.
# Hex values match configs/scalar_fields.xml exactly.
# Label order matches configs/label_map.json: 9=other, 10=column, 11=arch.

CLASS_NAMES = [
    'wall', 'window', 'door', 'balcony', 'vegetation', 'stairs',
    'terrain', 'roof', 'blinds', 'other', 'column', 'arch',
]

HEX = [
    '#ff2828',  # 0  wall
    '#00ccff',  # 1  window
    '#005449',  # 2  door
    '#ffee00',  # 3  balcony
    '#00ff50',  # 4  vegetation
    '#ff99ff',  # 5  stairs
    '#b84600',  # 6  terrain
    '#ff00b4',  # 7  roof
    '#7800ff',  # 8  blinds
    '#a0a0a0',  # 9  other
    '#4f0004',  # 10 column
    '#000000',  # 11 arch
]

def _hex_to_rgb_norm(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

RGB_NORM = [_hex_to_rgb_norm(h) for h in HEX]

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
