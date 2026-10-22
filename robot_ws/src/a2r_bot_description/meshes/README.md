# meshes/ — optional high-fidelity bodywork

The model in `urdf/` is built entirely from URDF primitives (boxes and
cylinders). That is a deliberate default:

* it renders instantly in RViz on a Raspberry Pi,
* it needs zero binary assets in the repo,
* it never breaks because of a missing or mis-scaled file.

A ~123-link primitive body gets you a sharp, readable vehicle. It will not get
you reflections, bevels or curved surfaces, because **RViz shades URDF
materials as flat RGBA** — there is no specular, metalness or roughness term
in the URDF material spec at all. Studio-render quality needs real meshes.

## Dropping in real bodywork

1. Export your body as `body.dae` (COLLADA, preferred — it carries per-face
   materials) or `body.stl` (single colour only).
2. Put it in this folder.
3. Model in **metres**, Z up, X forward, with the origin at the point that
   should sit at `base_link` + 0.03 m.
4. Launch with meshes enabled:

   ```bash
   ros2 launch a2r_bot_description display.launch.py use_meshes:=true
   ```

The primitives stay in place as the fallback, so `use_meshes:=false` always
works and CI never depends on an asset being present.

## Keeping collision cheap

Only ever attach meshes to **visual** geometry. `base_link`'s collision box
stays a primitive: the local planner's footprint, the inflation radius and the
Gazebo contact solver all want a convex primitive, and a 200k-triangle body
shell would slow every one of them down for no benefit.

## Suggested per-part files

If you want to swap parts individually rather than one monolithic shell:

| File | Replaces |
|---|---|
| `body.dae` | the 30 styling boxes in `chassis.xacro` |
| `wheel.dae` | tyre + rim + spokes in `wheels.xacro` |
| `lidar.dae` | `lidar_link` visual |
| `jetson.dae`, `rpi4.dae` | deck electronics |

Then add a `<mesh>` visual to the relevant link and guard it with the existing
`use_meshes` xacro property.
