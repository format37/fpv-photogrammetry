# Capturing footage for reconstruction

Lessons from four scenes shot with an FPV wing and a DJI O4 Pro. Two worked well, two did not.

## Camera

- **Fixed field of view.** Use the normal (rectilinear) mode. Ultra-wide modes need a fisheye model, give
  fewer pixels per degree, and reconstructed visibly worse.
- **Highest resolution; frame rate does not matter.** Keyframes are about one per second.
- **Short exposure.** Motion blur kills feature matching: use an ND filter only as strong as needed and
  keep the shutter fast.
- **Fixed exposure and white balance** if the camera allows it.
- **Stabilisation.** Electronic stabilisation changes the effective intrinsics slightly from frame to
  frame. It still works (0.75 px), but off is better if the footage stays usable.
- **Tilt the camera 30–45° down.** A forward-looking camera sees distant terrain at a grazing angle:
  depth noise then smears along the viewing rays and the ground comes out soft.
- **Start recording on arm**, so video time and log time share a zero.

## Flight

- **Height: 1–3× the size of the detail you want.** A site seen only from 5× farther than the flying
  height came out as noise.
- **Several parallel, offset passes** (and a crossing one) over the object, not one fly-by. Parallax
  comes from sideways offset between passes, not from flying towards the object.
- **Overlap between passes.** Each pass must see ground the others see. A pass with little shared
  texture can be reconstructed mirrored or as a separate model.
- **Smooth and level.** Hard banking turns give blur and sky-only frames.
- **No high overview passes.** They add frames without parallax and do not help.
- **Speed.** 15–20 m/s at 150–400 m above the terrain was fine; lower needs slower.
- Orbiting an object is the best single pattern for it.

## Light and weather

- **Avoid haze.** Distant slopes lose texture and produce no depth.
- **Never fly a pass into the sun.** Flare and glare frames do not register.
- **Low sun is a trade.** Long shadows give texture but deep shadow gives holes, and shadows move:
  keep the whole capture under about 20 minutes.
- **Best:** bright overcast or mid-morning/mid-afternoon sun behind the camera.
- **Poor subjects:** water, snow, uniform grass, moving vegetation.

## Telemetry

- **Log GPS at 5 Hz or more** with a good fix before arming (20+ satellites).
- Do not move the aircraft between arming and launch if the home point matters.
- A camera-only model has no scale. Without GPS you need two points with a known distance.
- For survey-grade results: ground control points or RTK, and a downward camera on a grid.
