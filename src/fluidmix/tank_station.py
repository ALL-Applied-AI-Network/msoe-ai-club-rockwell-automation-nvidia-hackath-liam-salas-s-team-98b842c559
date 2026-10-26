"""Procedural dosing tanks with explicit synthetic dimensions and volume keyframes.

Geometry is a single-view approximation, not measured plant CAD. No fluid physics,
carryover model, or camera calibration is implied by this asset.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


TANK_IDS = ("T100", "T200", "T300", "T400")
ROOT = "/TankStation"
COLORS = {"T100": (0.48, 0.58, 0.53), "T200": (0.44, 0.17, 0.21),
          "T300": (0.5, 0.5, 0.5), "T400": (0.055, 0.16, 0.26)}


@dataclass(frozen=True)
class TankGeometry:
    """Assumed dimensions in metres; liquid volume is expressed in mL."""

    radius: float = 0.07
    body_height: float = 0.16
    cone_height: float = 0.10

    def __post_init__(self):
        for value in (self.radius, self.body_height, self.cone_height):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("tank dimensions must be finite and positive")

    @property
    def cone_ml(self):
        return math.pi * self.radius**2 * self.cone_height * 1e6 / 3

    @property
    def capacity_ml(self):
        return self.cone_ml + math.pi * self.radius**2 * self.body_height * 1e6

    def height_for_volume(self, volume_ml):
        """Return height from the cone tip, rejecting impossible volumes."""
        if not math.isfinite(volume_ml) or not 0 <= volume_ml <= self.capacity_ml:
            raise ValueError(f"volume must be in [0, {self.capacity_ml:.6f}] mL")
        if volume_ml <= self.cone_ml:
            return self.cone_height * (volume_ml / self.cone_ml) ** (1 / 3)
        return self.cone_height + (volume_ml - self.cone_ml) / (math.pi * self.radius**2 * 1e6)


def _label(prim, category, instance):
    for name, value, kind in (("synthetic_class", category, Sdf.ValueTypeNames.String),
                              ("synthetic_instance_id", instance, Sdf.ValueTypeNames.Int)):
        prim.CreateAttribute("userProperties:" + name, kind).Set(value)


def _material(stage, name, color, opacity=1.0, metallic=0.0, roughness=0.4):
    mat = UsdShade.Material.Define(stage, ROOT + "/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    for key, kind, value in (("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
                             ("opacity", Sdf.ValueTypeNames.Float, opacity),
                             ("metallic", Sdf.ValueTypeNames.Float, metallic),
                             ("roughness", Sdf.ValueTypeNames.Float, roughness),
                             ("ior", Sdf.ValueTypeNames.Float, 1.333)):
        shader.CreateInput(key, kind).Set(value)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _box(stage, path, center, size, material, category="structure", instance=0):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(shape.GetPrim(), material)
    _label(shape.GetPrim(), category, instance)
    return shape


def _rod(stage, path, start, end, radius, material):
    a, b = Gf.Vec3d(*start), Gf.Vec3d(*end)
    direction = b - a
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(direction.GetLength())
    cylinder.CreateAxisAttr("Z")
    cylinder.AddTranslateOp().Set((a + b) / 2)
    cylinder.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Rotation(Gf.Vec3d(0, 0, 1), direction.GetNormalized()).GetQuat())
    _bind(cylinder.GetPrim(), material)
    _label(cylinder.GetPrim(), "plumbing", 0)
    return cylinder


def _profile_points(geometry, height, segments=64):
    # Fixed topology at all times. Within the cone, the middle and top rings
    # coincide; above it the middle ring stays at the cone/body junction.
    join = min(height, geometry.cone_height)
    radius = geometry.radius * min(1.0, height / geometry.cone_height)
    rings = ((0.0, 0.0), (radius, join), (radius, height))
    return [(r * math.cos(2 * math.pi * i / segments),
             r * math.sin(2 * math.pi * i / segments), z)
            for r, z in rings for i in range(segments)]


def _vessel(stage, path, geometry, material, capped=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    segments = 64
    faces = []
    for ring in range(2):
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([ring * segments + i, ring * segments + j,
                          (ring + 1) * segments + j, (ring + 1) * segments + i])
    if capped:
        faces.append(list(range(2 * segments, 3 * segments)))
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr(_profile_points(geometry, geometry.cone_height + geometry.body_height))
    _bind(mesh.GetPrim(), material)
    return mesh


def set_volumes(stage, volumes, frame, root=ROOT):
    """Author one complete volume sample, validating before mutating the stage.

    Args:
        stage: Writable USD stage containing the generated tank station.
        volumes: Mapping of all four uppercase tank IDs to mL. T300 must be zero.
        frame: Finite USD time code within the stage range.
        root: Station prim path, including a composition prefix if applicable.
    """
    if set(volumes) != set(TANK_IDS):
        raise ValueError("supply exactly T100, T200, T300 and T400 volumes")
    if not math.isfinite(frame) or not stage.GetStartTimeCode() <= frame <= stage.GetEndTimeCode():
        raise ValueError("frame outside the stage timeline")
    if volumes["T300"] != 0:
        raise ValueError("T300 is out of service and must remain empty")
    validated = []
    for tank in TANK_IDS:
        prim = stage.GetPrimAtPath(root + "/" + tank)
        if not prim:
            raise ValueError(f"missing tank {tank}")
        geo = TankGeometry(*(prim.GetAttribute("fluidmix:" + key).Get()
                             for key in ("radius", "bodyHeight", "coneHeight")))
        volume = float(volumes[tank])
        height = geo.height_for_volume(volume)
        validated.append((prim, geo, volume, height))
    for prim, geo, volume, height in validated:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(str(prim.GetPath()) + "/Liquid"))
        mesh.GetPointsAttr().Set(_profile_points(geo, height), frame)
        mesh.CreateExtentAttr().Set([(-geo.radius, -geo.radius, 0),
                                    (geo.radius, geo.radius, height)], frame)
        mesh.CreateVisibilityAttr().Set("invisible" if volume == 0 else "inherited", frame)
        prim.GetAttribute("fluidmix:volumeMl").Set(volume, frame)


def set_colors(stage, colors, root=ROOT):
    """Set per-tank linear RGB appearance, without claiming optical calibration."""
    validated = []
    for tank, color in colors.items():
        if tank not in TANK_IDS or len(color) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in color):
            raise ValueError("colors require known tank IDs and three finite values in [0,1]")
        shader = UsdShade.Shader(stage.GetPrimAtPath(root+"/Materials/"+tank+"Liquid/Surface"))
        if not shader:
            raise ValueError(f"missing liquid material for {tank}")
        validated.append((shader, color))
    for shader, color in validated:
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))


def _nameplate(stage, path, text, center, instance):
    """Build readable geometry labels without renderer-specific texture graphs."""
    plate = _material(stage, "Label"+str(instance), (.82,.85,.78))
    ink = _material(stage, "LabelInk"+str(instance), (.015,.025,.03))
    label = UsdGeom.Xform.Define(stage,path)
    label.AddTranslateOp().Set(Gf.Vec3d(*center))
    label.AddScaleOp().Set(Gf.Vec3f(.24))
    x,y,z = 0,0,0
    _box(stage, path+"/Plate", (0,0,0), (.115,.003,.045), plate,"tank_label",instance)
    segments = {
        "0":"abcdef", "1":"bc", "2":"abdeg", "3":"abcdg", "4":"bcfg",
        "O":"abcdef", "F":"aefg",
    }
    layout = {"a":(0,.009,.014,.002),"b":(.007,.0045,.002,.009),
              "c":(.007,-.0045,.002,.009),"d":(0,-.009,.014,.002),
              "e":(-.007,-.0045,.002,.009),"f":(-.007,.0045,.002,.009),
              "g":(0,0,.014,.002)}
    lines = [(text.split()[0],0,1)]
    if "OFF" in text:
        lines.append(("OFF",-.017,.45))
    for line_index,(line,zoff,scale) in enumerate(lines):
        for i,char in enumerate(line):
            cx = x+(i-(len(line)-1)/2)*.024*scale
            strokes = [(0,.009,.014,.002),(0,0,.002,.018)] if char == "T" else [layout[k] for k in segments[char]]
            for j,(dx,dz,width,height) in enumerate(strokes):
                _box(stage,path+f"/Glyph{line_index}_{i}_{j}",
                     (cx+dx*scale,y-.0025,z+zoff+dz*scale),
                     (width*scale,.001,height*scale),ink,"tank_label",instance)


def build_station(path, geometry=TankGeometry(), samples=None, end_frame=686):
    """Write a standalone tank station; refuse to overwrite an existing asset."""
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    if not math.isfinite(end_frame) or end_frame <= 1 or end_frame != int(end_frame):
        raise ValueError("end_frame must exceed 1")
    if samples is None:
        samples = [(1, {t: geometry.capacity_ml * f for t, f in zip(TANK_IDS, (.82,.75,0,.85))}),
                   (end_frame, {t: geometry.capacity_ml * f for t, f in zip(TANK_IDS, (.35,.25,0,.4))})]
        sample_source = "illustrative drawdown; not recipes or measured flow"
    else:
        sample_source = "caller supplied volume keyframes; source must be recorded by caller"
    # Sample each integer frame after linear interpolation of volumes: interpolating
    # mesh height alone inside a cone would violate the volume/height relationship.
    samples = sorted(samples, key=lambda sample: sample[0])
    if len(samples) < 2 or samples[0][0] != 1 or samples[-1][0] != end_frame:
        raise ValueError("samples must cover frame 1 through end_frame")
    for index, (frame, volumes) in enumerate(samples):
        if not math.isfinite(frame) or frame != int(frame):
            raise ValueError("sample frames must be finite integers")
        if index and frame <= samples[index-1][0]:
            raise ValueError("duplicate or unordered frame")
        if set(volumes) != set(TANK_IDS) or volumes["T300"] != 0:
            raise ValueError("four tank volumes required; T300 must be empty")
        for volume in volumes.values():
            geometry.height_for_volume(volume)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, ROOT).GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(end_frame)
    stage.SetTimeCodesPerSecond(30)
    stage.SetFramesPerSecond(30)
    stage.GetRootLayer().customLayerData = {
        "fluidmix:geometry": "single-view approximation; dimensions not measured",
        "fluidmix:animation": sample_source,
        "fluidmix:optics": "uncalibrated UsdPreviewSurface coverage; no absorption physics",
    }
    steel = _material(stage, "Steel", (.5,.55,.58), metallic=.8, roughness=.27)
    white = _material(stage, "FrostedPolymer", (.70,.74,.65), opacity=.42, roughness=.65)
    panel = _material(stage, "Panel", (.63,.67,.61), metallic=.4)
    dark = _material(stage, "Frame", (.035,.055,.075), metallic=.5)
    _box(stage, ROOT+"/Backboard", (0,.13,.62), (.96,.018,.74), panel)
    _box(stage, ROOT+"/Base", (0,0,.245), (1.03,.34,.025), steel)
    for x in (-.49,.49):
        _box(stage, ROOT+f"/Upright{'L' if x < 0 else 'R'}", (x,.02,.63), (.018,.026,.78), steel)
    for index, z in enumerate((.28,.79,.83)):
        _rod(stage, ROOT+f"/Rail{index}", (-.49,0,z), (.49,0,z), .009, steel)
    bottom = .31
    top = bottom + geometry.cone_height + geometry.body_height
    for index, tank in enumerate(TANK_IDS):
        x = (index - 1.5) * .235
        base = ROOT + "/" + tank
        prim = UsdGeom.Xform.Define(stage, base)
        prim.AddTranslateOp().Set(Gf.Vec3d(x,0,bottom))
        for key, val in (("radius",geometry.radius), ("bodyHeight",geometry.body_height),
                         ("coneHeight",geometry.cone_height), ("capacityMl",geometry.capacity_ml),
                         ("volumeMl",0.0)):
            prim.GetPrim().CreateAttribute("fluidmix:"+key,Sdf.ValueTypeNames.Double).Set(val)
        prim.GetPrim().CreateAttribute("fluidmix:active",Sdf.ValueTypeNames.Bool).Set(tank != "T300")
        shell = _vessel(stage, base+"/Shell", geometry, white, capped=False)
        shell.AddScaleOp().Set(Gf.Vec3f(1.025,1.025,1))
        # RTX transparency annotations: let the visible liquid receive labels
        # through the shell. This also disables shell shadows, an optical limit.
        UsdGeom.PrimvarsAPI(shell).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        _label(shell.GetPrim(), "tank", index+1)
        material = _material(stage, tank+"Liquid", COLORS[tank], roughness=.32)
        liquid = _vessel(stage, base+"/Liquid", geometry, material)
        _label(liquid.GetPrim(), "liquid", index+1)
        _rod(stage, base+"/Lid", (0,0,top-bottom), (0,0,top-bottom+.012), geometry.radius+.004, steel)
        _rod(stage, base+"/Outlet", (0,0,-.055), (0,0,.012), .009, steel)
        _box(stage, base+"/Valve", (0,-.012,-.045), (.044,.018,.012), dark)
        for side in (-1,1):
            _rod(stage, base+f"/Leg{'L' if side<0 else 'R'}", (side*.057,-.025,-.05),
                 (side*.057,-.025,.15), .006, steel)
            _rod(stage, base+f"/Inlet{'L' if side<0 else 'R'}", (side*.035,0,top-bottom),
                 (side*.035,.01,.48), .0045, steel)
            _rod(stage, base+f"/Fitting{'L' if side<0 else 'R'}", (side*.035,0,top-bottom+.01),
                 (side*.035,0,top-bottom+.045), .010, steel)
        _nameplate(stage, base+"/Label", tank if tank != "T300" else "T300 OFF",
                   (0,-geometry.radius-.004,top-bottom-.025), index+1)
        _box(stage, base+"/LevelStrip", (.025,-geometry.radius-.005,.075), (.016,.003,.18), panel,
             "level_scale", index+1)
        for tick in range(10):
            _box(stage, base+f"/Tick{tick}", (.025,-geometry.radius-.007,-.01+tick*.019),
                 (.009,.002,.0015), dark, "level_scale", index+1)
    for (start, old), (end, new) in zip(samples, samples[1:]):
        for frame in range(int(start), int(end)+1):
            alpha = (frame-start)/(end-start)
            set_volumes(stage, {t: old[t]+alpha*(new[t]-old[t]) for t in TANK_IDS}, frame)
    camera = UsdGeom.Camera.Define(stage, ROOT+"/TankCam")
    camera.CreateFocalLengthAttr(35)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(27)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    # Fixed-view registration against manually selected landmarks. Assumed scale
    # and pinhole lens: this is not a recovered physical camera calibration.
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-.100136,-1.365533,.3),
                                 Gf.Vec3d(-.127791,0,.755748), Gf.Vec3d(-.002291,0,.999997))
    camera.AddTransformOp().Set(view.GetInverse())
    dome = UsdLux.DomeLight.Define(stage, ROOT+"/Dome")
    dome.CreateIntensityAttr(450)
    light = UsdLux.RectLight.Define(stage, ROOT+"/Softbox")
    light.CreateIntensityAttr(650)
    light.CreateWidthAttr(1.2)
    light.CreateHeightAttr(.8)
    light.AddTranslateOp().Set(Gf.Vec3d(0,-.45,1.2))
    stage.GetRootLayer().Save()
    return stage


def compose_line(station_path, twin_path, output_path, position=(0,0,0)):
    """Reference the unchanged full line and add tanks in fill-station coordinates.

    Placement is explicit and uncalibrated. The supplied offset is in metres in
    the fill station's local frame; it is not a measured tank/nozzle relationship.
    """
    station_path, twin_path, output_path = map(lambda p: Path(p).resolve(),
                                              (station_path,twin_path,output_path))
    line_path = twin_path / "csi_line.usda"
    if output_path.exists():
        raise FileExistsError(output_path)
    source = Usd.Stage.Open(str(line_path))
    station = Usd.Stage.Open(str(station_path))
    if not source or not station or not source.GetPrimAtPath("/CSI_Line/FillStation"):
        raise ValueError("source line or tank station is invalid")
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        raise ValueError("position must contain three finite coordinates")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(line_path, output_path.parent))
    UsdGeom.SetStageUpAxis(stage,"Z")
    UsdGeom.SetStageMetersPerUnit(stage,1)
    for setter, value in ((stage.SetStartTimeCode,source.GetStartTimeCode()),
                          (stage.SetEndTimeCode,max(source.GetEndTimeCode(),station.GetEndTimeCode())),
                          (stage.SetTimeCodesPerSecond,source.GetTimeCodesPerSecond()),
                          (stage.SetFramesPerSecond,source.GetFramesPerSecond())):
        setter(value)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/CSI_Line"))
    placement = UsdGeom.Xform.Define(stage,"/CSI_Line/TankPlacement")
    matrix = UsdGeom.Xformable(source.GetPrimAtPath("/CSI_Line/FillStation")).ComputeLocalToWorldTransform(1)
    placement.AddTransformOp().Set(matrix)
    tanks = UsdGeom.Xform.Define(stage,"/CSI_Line/TankPlacement/TankStation")
    tanks.GetPrim().GetReferences().AddReference(os.path.relpath(station_path, output_path.parent))
    tanks.AddTranslateOp().Set(Gf.Vec3d(*position))
    stage.GetRootLayer().customLayerData = {"fluidmix:placement": "uncalibrated offset in fill-station frame"}
    stage.GetRootLayer().Save()
    return stage


def export_snapshot(stage, path, frame=1):
    """Export a self-contained static station with values frozen at one frame.

    Args:
        stage: Generated standalone stage to snapshot without modifying it.
        path: Fresh output USD path.
        frame: Integer frame within the source timeline.
    """
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    if not math.isfinite(frame) or frame != int(frame) or not stage.GetStartTimeCode() <= frame <= stage.GetEndTimeCode():
        raise ValueError("snapshot requires an integer frame within the timeline")
    frozen = Usd.Stage.Open(stage.Flatten(addSourceFileComment=False))
    for prim in frozen.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetNumTimeSamples():
                value = attr.Get(frame)
                attr.Clear()
                attr.Set(value)
    metadata = dict(frozen.GetRootLayer().customLayerData)
    metadata["fluidmix:animation"] = f"static synthetic snapshot at frame {frame}"
    frozen.GetRootLayer().customLayerData = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen.GetRootLayer().Export(str(path))
    if path.suffix == ".usda":
        path.write_text(path.read_text().rstrip()+"\n")
