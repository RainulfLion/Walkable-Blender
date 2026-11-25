# Castle art

The pictures on the walls and the banners of the castle come from this
folder. To change one, replace its file with your own image **using the same name**,
then reload the walkthrough (GitHub Pages can take a few minutes to pick it up).

| File | Where it shows | Shape |
| --- | --- | --- |
| `fresco1` | Great hall, north wall (far left); east wall | square, e.g. 1024 x 1024 |
| `fresco2` | Great hall, north wall; west wall | square |
| `fresco3` | Great hall, north wall; west wall | square |
| `fresco4` | Great hall, north wall (far right); throne room | square |
| `banner_red` | Gatehouse, great hall front, throne room, tall tower | tall, 1 : 2.4, e.g. 512 x 1229 |
| `banner_blue` | Gatehouse, great hall front, throne room | tall, 1 : 2.4 |

Notes:

- `.png`, `.jpg` and `.webp` all work. If two files share a name, `.png` wins, then
  `.jpg`, then `.webp`; delete the old one to avoid confusion.
- Images that are not the listed shape are stretched to fit.
- Banners hang from the top of the image. The bottom is cut to a point, so the two
  bottom corners are not visible; keep important detail out of them.
- Keep files under about 1 MB each so the level stays quick to load.
- The files here were generated as placeholders by `blender/build_castle.py`. That
  script never overwrites a file that is already in this folder, and it bakes
  whatever is here into `Castle.glb` and the `.blend` the next time it runs.

## Occidentals crest

`banner_red`, `banner_blue` and `fresco4` are made from `source/occidentals_crest.jpg`
by `blender/make_crest_art.py`. Swap the source image and re-run that script to
rebuild all three, or just replace any of the three files directly.
