"""Window colors for visualizing accepted image alignment correspondences."""
import colorsys
import math

import numpy as np


def window_color_labels(regions, physical):
    """Assign each masked physical window a color shared by all its matches.

    Connected components of the accepted bipartite window graph handle repeats:
    A0 matching B0 and B1 makes all three share one color. Unrelated windows,
    including gap-filled windows without a correspondence, get separate colors.
    """
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != node:
            previous = parents[node]
            parents[node] = root
            node = previous
        return root

    for region in regions:
        for i, j in region['pairs']:
            a = (0, int(physical[0][i]))
            b = (1, int(physical[1][j]))
            parents[find(b)] = find(a)
        for side in (0, 1):
            for key in ('supported_physical', 'filled_physical'):
                for index in region.get(key, ([], []))[side]:
                    find((side, int(index)))

    labels = [{}, {}]
    colors = {}
    for side in (0, 1):
        # Logical order preserves RTL/filtering; filled windows may not have a
        # feature row, but must still retain their own visible mask color.
        indices = list(map(int, physical[side]))
        indices += sorted(index for s, index in parents if s == side and index not in indices)
        for index in indices:
            node = (side, index)
            if node not in parents:
                continue
            root = find(node)
            if root not in colors:
                colors[root] = len(colors)
            labels[side][index] = colors[root]
    return labels


def aligned_window_overlays(originals, masks, regions, physical, geometry,
                            window_width, stride, alpha=.4):
    """Tint existing masks by window, with equal colors for matched windows.

    Source intervals use the exact crop/resize geometry. Where sliding windows
    overlap, the closest window center owns the pixel, so neighboring colors
    remain distinct instead of mixing or being overwritten by drawing order.
    Only supported or explicitly gap-filled windows are drawn. Binary masks,
    predictions and match scores are never modified.
    """
    labels = window_color_labels(regions, physical)
    count = max((color for side in labels for color in side.values()), default=-1) + 1
    # Golden-angle hues avoid recycling a short categorical palette.
    palette = np.asarray([colorsys.hsv_to_rgb((.04 + i * .618033988749895) % 1, .8, 1.)
                          for i in range(count)]) * 255
    overlays = []
    for image, mask, geo, colors in zip(originals, masks, geometry, labels):
        pixels = np.asarray(image.convert('RGB'), dtype=float).copy()
        owner = np.full(image.width, -1, dtype=int)
        distance = np.full(image.width, np.inf)
        x0, _, x1, _ = geo['crop']
        for index, color in sorted(colors.items()):
            start = max(x0, x0 + index * stride / geo['scale_x'])
            end = min(x1, x0 + (index * stride + window_width) / geo['scale_x'])
            left, right = max(0, math.floor(start)), min(image.width, math.ceil(end))
            if right <= left:
                continue
            delta = np.abs(np.arange(left, right) + .5 - (start + end) / 2)
            closer = delta < distance[left:right]
            distance[left:right][closer] = delta[closer]
            owner[left:right][closer] = color
        active = (np.asarray(mask) > 0) & (owner[None, :] >= 0)
        if count and active.any():
            tint = np.broadcast_to(palette[np.maximum(owner, 0)], pixels.shape)
            pixels[active] = pixels[active] * (1 - alpha) + tint[active] * alpha
        overlays.append(pixels.astype(np.uint8))
    return overlays
