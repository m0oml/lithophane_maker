# Lithophane Maker

Desktop app that turns a photo into a 3D-printable lithophane STL.

- Shapes: flat panel, curved arc (picture on the outside or inside), circle with optional hook
- Beveled frame (thickness, depth, angle), min/max thickness, detail in mm per pixel
- Live 3D preview with a backlit view
- Optional stand: extends the frame's bottom edge into a tab and saves a matching
  `<name>_stand.stl` with a slot (clearance, wall, footprint and 45° chamfer settings)

## Install and run

Needs Python 3.10+.

```sh
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
                                 # Windows: .venv\Scripts\activate
pip install -e .
lithophane
```
