# `corebreakout`

Mask-RCNN based segmentation of geological core images and assembly of depth-labeled image datasets.

Will include a general pre-trained model and tools for (re-)training with novel datasets.

Moving from "BlueLabel" painting-based labeling utilities to the more common polygon annotation format. Plenty of resources for polygons out there, and it's less finicky about exact colors and things. Probably should have just done it that way to begin with, but oh well.

*Question*: should processed core labeling tools (XML scripts, `striplog` figure tracks, etc.) be part of this package, or `coremdlr`?


### Additional object types for detection

One feature that could make this package more generally useful would be to have some options for using objects other than `core column` to determine the upper/lower pixel rows for cropping columns and generating accurate `depth` arrays.

I can see four basic options, each of which may work better or worse for any particular dataset:

- The hard-coded way we do it now. User specifies "layout(s)" that define the start/end of the core trays. This is a bit more of a hassle and probably sub-optimal, but it works well enough when the camera and tray positions are consistent.
- Taking the min/max of all the detected columns in an image. This would probably work well for times where you're pretty certain that at least one tray in each image will contain a full column worth of core, and there aren't shadows from tray edges.
- Detecting the core `box` as a seperate object, and using its bounding box to set the limits. This would work well for datasets with distinctive whole boxes (rather than e.g., partially occluded single trays), and where the core doesn't run over the edges of the boxes/trays too much.
- Detecting `scale` objects. This would similarly work well for datasets with distinctive scales that have relatively good contrast against the background (and that actually line up with the core trays).

It wouldn't be too hard to include these as options, especially since `box` and `scale` would use pretty much (if not exactly) the same logic. Ultimately then it would be up to the user to decide which option is best for their dataset and whether they want to spend time labeling those extra objects.


### Other TODO

- Label some  more varied data from `pretrained/data/` (using `labelme`) and test out the new `PolygonDataset` class

- Build small `train` and `test` sets with new data

- Put (re-)training utilities into a script, test it out with the new data

- Move (parts of) `CorePlotter` into this package for viz (check tick generation -- make sure it works with `mode='collapse'`)

- Start on a write up for JOSS submission (?)
