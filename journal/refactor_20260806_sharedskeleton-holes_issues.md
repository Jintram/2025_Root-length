

*Notes by Claude Opus 5 regarding validity of root+shoot mask approach.* 


# 2026-08-06 — shared root+shoot skeleton

Root and shoot used to be skeletonized separately, which distorts both
centerlines at the border between them (the medial axis of a truncated shape
retracts from the cut). They are now merged into one binary mask, skeletonized
once, and the resulting skeleton is divided again by assigning each pixel to the
nearest tissue. Holes (a seed interrupting the root) are bridged by dilating
with the smallest radius that makes the mask whole; if that radius exceeds
`dilation_radius_maximum` the plant falls back to the old per-tissue treatment.
See `prepare_shared_skeleton` in `determine_length.py`, and set
`ConfigPipeline(shared_skeleton=False)` to get the old behaviour back.


## Open issue: root and shoot overlapping in projection

Where the shoot mask lies alongside or across the root instead of meeting it
end to end, the union of the two masks is locally wide, so the shared skeleton
runs through shoot-labeled territory somewhere halfway down the root. Splitting
the skeleton then leaves the root in two or three disconnected parts, and only
the part yielding the longest path gets measured. The rest of the root length is
simply lost.

This shows up as `root_skeleton_parts > 1` (or `shoot_skeleton_parts > 1`) in
the `_lengths.tsv` output, which is the quickest way to find affected plants.
Note the problem is unrelated to the hole bridging: it also occurs at
`dilation_radius_used == 0`, i.e. when no dilation happened at all.

### Where this was seen

In `2025_10_hypocotyl-root-length/BACKUPS_TRAININGDATA/MODEL2/segfiles-corrected_YUZENGmanual/`,
comparing `shared_skeleton=True` against `False` over the first 6 segfiles
(131 plants), 5 plants lost more than 20 px of root length:

| segfile                       | plant_index | root old -> new | parts |
|-------------------------------|-------------|-----------------|-------|
| 20251104batch24_OY_10_seg.npz | 4           | 242.8 -> 189.7  | 2     |
| 20251104batch24_OY_10_seg.npz | 17          |  38.5 ->   8.8  | 2     |
| 20251104batch24_OY_12_seg.npz | 0           | 193.0 ->  89.7  | 3     |
| 20251104batch24_OY_18_seg.npz | 11          | 218.6 -> 157.4  | 2     |
| 20251107batch24_OY_38_seg.npz | 1           |  65.3 ->  22.9  | 3     |

(`plant_index` counts the QC-passed plants of that plate, i.e. it matches the
column of the same name in the `.tsv`, and `list_img_indivplants[sel_plants]`
in `analyze_plate`.)

Over the same 131 plants the overall effect was small and centered on zero
(root: mean -2.4 px, median 0; shoot: mean +2.5 px, median 0), with 25 plants
bridged and 8 falling back, so these 5 are the tail, not the norm.

### Why it is parked

This segmentation data has not been corrected by hand yet and contains
artifacts that should not survive correction — shoot label sitting on top of, or
next to, the root. Revisit once corrected data is available: rerun the
comparison above and check whether `root_skeleton_parts > 1` still occurs.

### What currently happens with a disconnected tissue graph

Splitting the shared skeleton per tissue can leave a tissue in several parts, so
`sample.segment_graph` genuinely has multiple connected components. Note this is
a different level than a hole in the *mask*: holes are dealt with by the
dilation beforehand (a root cut in two by a seed comes back connected, because
the bridge pixels are assigned to the nearest tissue, i.e. the root), and if that
fails the plant goes to the fallback instead.

Current handling: `find_start_labels_close_to_anchor` returns one start label per
component (the segment of that component closest to the other tissue), and
`get_long_path_in_graph_nodearea` restricts its targets to
`nx.node_connected_component(graph, source)`. So the search runs per component
and the **single best component wins** — the length in all other components is
silently dropped.

The per-component start labels matter a lot here. With one global start label
(the earlier version), a 1-3 px stray fragment that happened to sit closest to
the other tissue would win the start and trap the search inside itself, giving
root lengths of ~4 px instead of ~270 px. That hit 10 of the 131 test plants.

### Idea: sum over components instead of taking only the largest

Taking only the best component is what loses the length in the table above. The
parts are all genuinely the same root, so **summing the longest path of each
component** should recover most of it, and is probably the better fix — simpler
than the redesign below and it also covers the case where a tissue is split for
reasons other than an overlap.

Points to keep in mind when implementing this:

- The stretch that was cut out (where the skeleton ran through the other
  tissue's territory) is still not counted, so it stays a slight underestimate,
  but a much smaller one than dropping a whole component.
- Noise fragments get added too. A length threshold below which a component is
  ignored would probably be wanted, otherwise every stray pixel adds ~0.5-3 px.
- The semantics become mixed: *max* within a component, *sum* across components.
  Defensible, but worth being explicit about, since a spurious branch that ends
  up as its own component is then added rather than suppressed the way branches
  within a component are.
- Implementation gotcha: do not compute this by unioning the per-component path
  masks and calling `get_length_longestpath` once. `get_length_segment` warns and
  returns `nan` as soon as any pixel in the mask has fewer than 1 neighbor, which
  an isolated single-pixel component would trigger. Compute the length per
  component and add them up.

### If it does turn out to be real

The more structural fix is to build **one** graph over the whole shared skeleton
instead of one per tissue: split segments at the tissue border so each segment is
purely root or purely shoot, run a single shortest-path pass from the junction
segment, and take root length = furthest root-labeled node, shoot length =
furthest shoot-labeled node. A root path may then traverse a short shoot stretch
and continue, instead of being cut in two by it. That also removes the
disconnected-graph handling in `get_long_path_in_graph_nodearea`, and replaces
the all-pairs loop there with one single-source pass.


## Side note: why dilation and not closing

An earlier attempt bridged holes with a morphological closing. That needs a
radius far larger than half the gap, which was confusing at first: dilation
merges two regions separated by a gap g once both grow by g/2, but the erosion
step then removes the bridge again, because the erosion disk can reach into the
gap *around* the tips of a thin structure. For two bars of width w facing each
other across gap g the closing only fills the gap once

    r > g^2/(4w) + w/4

(the disk that rolls over both tips; sagitta relation). For g = w this reduces to
the naive g/2, but for g = 4w it is about 1.06*g, which is why `r = g+1`
appeared to work empirically. It is not a safe general rule — at g = 6w it needs
about 1.5*g. Dilation without the erosion step has none of this: g/2 is correct,
the radius stays small, and small matters because dilation straightens bends
tighter than its radius and fuses structures passing within twice its radius.
The surplus material the dilation adds at the free ends (root tip, shoot apex)
is removed afterwards by `prune_skeleton_outside_mask`.
