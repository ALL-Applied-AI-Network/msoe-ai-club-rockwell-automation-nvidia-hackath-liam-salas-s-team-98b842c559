"""Export the generated USD tanks to a self-contained WebGL playback page.

This is a geometry and animation preview, not an RTX or optical simulation.
Only the generated station's Mesh, Cube and Cylinder primitives are supported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdShade


def topology(prim, frame):
    """Return local points and polygon faces for a supported USD primitive."""
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(frame)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        faces, offset = [], 0
        for count in counts:
            faces.append(list(indices[offset:offset + count]))
            offset += count
        return points, faces
    if prim.IsA(UsdGeom.Cube):
        half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
        return [Gf.Vec3d(x*half, y*half, z*half) for x, y, z in
                [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                 (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]], [
                    [0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    if prim.IsA(UsdGeom.Cylinder):
        shape = UsdGeom.Cylinder(prim)
        if shape.GetAxisAttr().Get() != "Z":
            raise ValueError("preview supports only Z-axis cylinders")
        radius, half = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get()/2
        points = [Gf.Vec3d(radius*math.cos(i*math.tau/32), radius*math.sin(i*math.tau/32), z)
                  for z in (-half, half) for i in range(32)]
        faces = [[i,(i+1)%32,(i+1)%32+32,i+32] for i in range(32)]
        return points, faces + [list(reversed(range(32))), list(range(32,64))]
    raise ValueError(f"unsupported shape {prim.GetPath()}")


def export_data(stage_path):
    """Read geometry, materials, camera, and every volume sample from the USD."""
    stage = Usd.Stage.Open(str(stage_path))
    root = stage.GetPrimAtPath("/TankStation")
    if not root:
        raise ValueError("preview requires the standalone /TankStation asset")
    start, end = int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode())
    cache = UsdGeom.XformCache(start)
    objects = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        points, faces = topology(prim, start)
        matrix = cache.GetLocalToWorldTransform(prim)
        triangles = [index for face in faces for j in range(1,len(face)-1)
                     for index in (face[0],face[j],face[j+1])]
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shader = UsdShade.Shader(material.GetPrim().GetChild("Surface"))
        color = list(shader.GetInput("diffuseColor").Get())
        opacity = shader.GetInput("opacity").Get()
        obj = {"path":str(prim.GetPath()),"points":[list(matrix.Transform(Gf.Vec3d(p))) for p in points],
               "triangles":triangles,"color":color,"opacity":opacity}
        if prim.GetName() == "Liquid":
            tank = prim.GetParent()
            obj["tank"] = tank.GetName()
            obj["samples"] = [[list(p) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get(f)]
                              for f in range(start,end+1)]
            obj["origin"] = list(matrix.Transform(Gf.Vec3d(0)))
            obj["volumes"] = [tank.GetAttribute("fluidmix:volumeMl").Get(f) for f in range(start,end+1)]
            obj["capacity"] = tank.GetAttribute("fluidmix:capacityMl").Get()
        objects.append(obj)
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/TankStation/TankCam")).GetCamera(start)
    # Gf uses row vectors; flattening rows supplies the transpose WebGL expects.
    transform = camera.frustum.ComputeViewMatrix()*camera.frustum.ComputeProjectionMatrix()
    return {"objects":objects,"viewProjection":[v for row in transform for v in row],
            "start":start,"end":end,"fps":stage.GetTimeCodesPerSecond(),
            "source_sha256":hashlib.sha256(stage_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh output HTML path")
    data = export_data(args.stage)
    template = Path(__file__).with_name("tank_preview.html").read_text()
    encoded = json.dumps(data,separators=(",",":"))
    # Escape HTML delimiters even though all prim names originate from our asset.
    encoded = encoded.replace("<", "\\u003c")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(template.replace("__TANK_DATA__",encoded))
    print(json.dumps({"html":str(args.out.resolve()),"objects":len(data["objects"]),
                      "frames":data["end"]-data["start"]+1,"source_sha256":data["source_sha256"]}))


if __name__ == "__main__":
    main()
