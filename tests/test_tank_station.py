"""Synthetic asset checks; no plant or camera accuracy claims."""
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("pxr", reason="tank asset checks require optional usd-core")
from pxr import Gf, Usd, UsdGeom, UsdShade
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fluidmix.tank_station import TankGeometry, TANK_IDS, build_station, compose_line, set_colors, set_volumes, export_snapshot


def test_analytic_volume_height():
    geo = TankGeometry()
    assert geo.height_for_volume(0) == 0
    assert geo.height_for_volume(geo.cone_ml) == pytest.approx(geo.cone_height)
    assert geo.height_for_volume(geo.capacity_ml) == pytest.approx(geo.body_height+geo.cone_height)
    for volume in np.linspace(0,geo.capacity_ml,41):
        height = geo.height_for_volume(volume)
        if height <= geo.cone_height:
            recovered = math.pi*(geo.radius*height/geo.cone_height)**2*height*1e6/3
        else:
            recovered = geo.cone_ml+math.pi*geo.radius**2*(height-geo.cone_height)*1e6
        assert recovered == pytest.approx(volume)


@pytest.mark.parametrize("volume",[-1,float("nan"),float("inf"),4000])
def test_impossible_volume(volume):
    with pytest.raises(ValueError):
        TankGeometry().height_for_volume(volume)


@pytest.mark.parametrize("radius",[0,-.1,float("nan"),float("inf")])
def test_bad_dimensions(radius):
    with pytest.raises(ValueError):
        TankGeometry(radius=radius)


@pytest.fixture
def station(tmp_path):
    return build_station(tmp_path/"asset"/"tanks.usda",end_frame=5)


def test_saved_asset_and_materials(station):
    path = station.GetRootLayer().realPath
    reopened = Usd.Stage.Open(path)
    assert reopened.GetDefaultPrim().GetPath() == "/TankStation"
    assert UsdGeom.GetStageUpAxis(reopened) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(reopened) == 1
    assert reopened.GetFramesPerSecond() == reopened.GetTimeCodesPerSecond() == 30
    assert reopened.GetPrimAtPath("/TankStation/TankCam").IsA(UsdGeom.Camera)
    assert reopened.GetPrimAtPath("/TankStation/T100/Label/Glyph0_0_0")
    for tank in TANK_IDS:
        assert reopened.GetPrimAtPath(f"/TankStation/{tank}/Liquid").IsA(UsdGeom.Mesh)
        assert reopened.GetPrimAtPath(f"/TankStation/Materials/{tank}Liquid/Surface")
        assert reopened.GetPrimAtPath(f"/TankStation/{tank}/Shell").GetAttribute("primvars:doNotCastShadows").Get() is True
    assert reopened.GetPrimAtPath("/TankStation/T300").GetAttribute("fluidmix:active").Get() is False
    for frame in (1,3,5):
        assert UsdGeom.Imageable(reopened.GetPrimAtPath("/TankStation/T300/Liquid")).ComputeVisibility(frame) == "invisible"


def test_liquid_mesh_volume_and_animation(station):
    for frame in (1,3,5):
        for tank in ("T100","T200","T400"):
            prim = station.GetPrimAtPath(f"/TankStation/{tank}")
            mesh = UsdGeom.Mesh(station.GetPrimAtPath(f"/TankStation/{tank}/Liquid"))
            points = np.asarray(mesh.GetPointsAttr().Get(frame))
            # Polygonal 64-sided approximation must remain within 0.2% of analytic volume.
            signed = 0.0
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            offset = 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset:offset+count]
                for i in range(1,count-1):
                    a,b,c = points[[face[0],face[i],face[i+1]]].astype(float)
                    signed += np.dot(a,np.cross(b,c))/6
                offset += count
            assert abs(signed)*1e6 == pytest.approx(prim.GetAttribute("fluidmix:volumeMl").Get(frame),rel=.002)
    mesh = UsdGeom.Mesh(station.GetPrimAtPath("/TankStation/T200/Liquid"))
    assert np.asarray(mesh.GetPointsAttr().Get(1))[:,2].max() > np.asarray(mesh.GetPointsAttr().Get(5))[:,2].max()


def test_cone_interpolation_is_in_volume(tmp_path):
    geo = TankGeometry()
    samples = [(1,dict(zip(TANK_IDS,(0,0,0,0)))),
               (3,dict(zip(TANK_IDS,(geo.cone_ml,geo.cone_ml,0,geo.cone_ml))))]
    stage = build_station(tmp_path/"cone"/"tanks.usda",samples=samples,end_frame=3)
    points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath("/TankStation/T100/Liquid")).GetPointsAttr().Get(2))
    assert points[:,2].max() == pytest.approx(geo.height_for_volume(geo.cone_ml/2))
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/TankStation/T100/Liquid")).ComputeVisibility(1) == "invisible"


@pytest.mark.parametrize("bad",[{"T100":0},dict(zip(TANK_IDS,(0,0,1,0))),dict(zip(TANK_IDS,(0,0,0,-1)))])
def test_bad_update_is_atomic(station,bad):
    before = station.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        set_volumes(station,bad,2)
    assert station.GetRootLayer().ExportToString() == before


def test_colors_are_independent(station):
    set_colors(station,{"T200":(.1,.2,.3)})
    shader = UsdShade.Shader(station.GetPrimAtPath("/TankStation/Materials/T200Liquid/Surface"))
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((.1,.2,.3))
    with pytest.raises(ValueError):
        set_colors(station,{"T400":(1,2,3)})


def test_compose_keeps_source_and_timing(station,tmp_path):
    twin = tmp_path/"twin"
    twin.mkdir()
    line = Usd.Stage.CreateNew(str(twin/"csi_line.usda"))
    line.SetDefaultPrim(UsdGeom.Xform.Define(line,"/CSI_Line").GetPrim())
    fill = UsdGeom.Xform.Define(line,"/CSI_Line/FillStation")
    fill.AddTranslateOp().Set(Gf.Vec3d(3,4,0))
    line.SetStartTimeCode(1)
    line.SetEndTimeCode(686)
    line.SetTimeCodesPerSecond(30)
    line.SetFramesPerSecond(30)
    line.GetRootLayer().Save()
    digest = hashlib.sha256((twin/"csi_line.usda").read_bytes()).hexdigest()
    combined = compose_line(station.GetRootLayer().realPath,twin,tmp_path/"combined.usda",(-1,0,0))
    assert combined.GetPrimAtPath("/CSI_Line/TankPlacement/TankStation/TankCam")
    root = combined.GetPrimAtPath("/CSI_Line/TankPlacement/TankStation")
    translation = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(1).ExtractTranslation()
    assert tuple(translation) == pytest.approx((2,4,0))
    assert combined.GetTimeCodesPerSecond() == 30
    assert combined.GetEndTimeCode() == 686
    assert hashlib.sha256((twin/"csi_line.usda").read_bytes()).hexdigest() == digest
    assert not combined.GetCompositionErrors()


def test_no_overwrite(station):
    with pytest.raises(FileExistsError):
        build_station(station.GetRootLayer().realPath)


def test_binary_export_is_self_contained(station,tmp_path):
    from pxr import UsdUtils
    binary = Path(station.GetRootLayer().realPath).with_suffix(".usdc")
    station.GetRootLayer().Export(str(binary))
    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(binary))
    assert len(layers) == 1
    assert not assets and not missing


def test_browser_export_preserves_usd_samples(station):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from preview_tank_station import export_data
    data = export_data(Path(station.GetRootLayer().realPath))
    assert len(data["viewProjection"]) == 16
    assert data["start"] == 1 and data["end"] == 5
    liquids = {o["tank"]:o for o in data["objects"] if "tank" in o}
    assert set(liquids) == set(TANK_IDS)
    for tank, obj in liquids.items():
        mesh = UsdGeom.Mesh(station.GetPrimAtPath(f"/TankStation/{tank}/Liquid"))
        for frame in (1,3,5):
            assert np.asarray(obj["samples"][frame-1]) == pytest.approx(np.asarray(mesh.GetPointsAttr().Get(frame)))
            assert obj["volumes"][frame-1] == station.GetPrimAtPath(f"/TankStation/{tank}").GetAttribute("fluidmix:volumeMl").Get(frame)
    assert liquids["T300"]["volumes"] == [0]*5
    assert liquids["T200"]["volumes"][0] > liquids["T200"]["volumes"][-1]
    assert all(0 <= i < len(o["points"]) for o in data["objects"] for i in o["triangles"])


@pytest.mark.parametrize("samples",[[(1,{t:0 for t in TANK_IDS})],
    [(1,{t:0 for t in TANK_IDS}),(1,{t:0 for t in TANK_IDS}),(3,{t:0 for t in TANK_IDS})]])
def test_invalid_samples_do_not_create_output(tmp_path,samples):
    path = tmp_path/"invalid.usda"
    with pytest.raises(ValueError):
        build_station(path,samples=samples,end_frame=3)
    assert not path.exists()


def test_snapshot_matches_frame_and_preserves_source(station,tmp_path):
    from pxr import UsdUtils
    before = station.GetRootLayer().ExportToString()
    path = tmp_path/"snapshot.usda"
    export_snapshot(station,path,3)
    assert station.GetRootLayer().ExportToString() == before
    snapshot = Usd.Stage.Open(str(path))
    for prim in snapshot.Traverse():
        assert all(a.GetNumTimeSamples() == 0 for a in prim.GetAttributes())
    for tank in TANK_IDS:
        p = f"/TankStation/{tank}"
        assert snapshot.GetPrimAtPath(p).GetAttribute("fluidmix:volumeMl").Get() == station.GetPrimAtPath(p).GetAttribute("fluidmix:volumeMl").Get(3)
        assert np.asarray(snapshot.GetPrimAtPath(p+"/Liquid").GetAttribute("points").Get()) == pytest.approx(np.asarray(station.GetPrimAtPath(p+"/Liquid").GetAttribute("points").Get(3)))
    layers,assets,missing = UsdUtils.ComputeAllDependencies(str(path))
    assert len(layers) == 1 and not assets and not missing
    with pytest.raises(FileExistsError):
        export_snapshot(station,path)


def test_committed_asset_rebuilds_exactly(tmp_path):
    asset = Path(__file__).resolve().parents[1]/"assets/tanks/tank_station.usda"
    stage = build_station(tmp_path/"animated.usda")
    export_snapshot(stage,tmp_path/"snapshot.usda")
    assert (tmp_path/"snapshot.usda").read_bytes() == asset.read_bytes()


def test_fixed_camera_reference_registration():
    import json
    root = Path(__file__).resolve().parents[1]
    landmarks = json.loads((root/'assets/tanks/reference-landmarks.json').read_text())
    stage = Usd.Stage.Open(str(root/'assets/tanks/tank_station.usda'))
    camera = UsdGeom.Camera(stage.GetPrimAtPath('/TankStation/TankCam')).GetCamera()
    transform = camera.frustum.ComputeViewMatrix()*camera.frustum.ComputeProjectionMatrix()
    errors = []
    for point in landmarks['points']:
        projected = transform.Transform(Gf.Vec3d(*point['world_m']))
        pixel = np.array([(projected[0]+1)*320,(1-projected[1])*240])
        errors.extend(pixel-np.array(point['pixel_xy']))
    # A fit-regression check on the same manually picked landmarks, not a
    # held-out accuracy claim or a recovered camera calibration.
    assert np.sqrt(np.mean(np.square(errors))) < 4.5


def test_capture_validator_rejects_corrupt_labels(station,tmp_path):
    import json
    from PIL import Image
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from check_tank_capture import check_capture
    folder = tmp_path/"capture"
    folder.mkdir()
    image_path = folder/"0001.00.png"
    rgb = np.zeros((480,640,3),dtype="uint8")
    rgb[:200] = 200
    Image.fromarray(rgb).save(image_path)
    instances = np.zeros((480,640),dtype="uint8")
    tanks, volumes = {}, {}
    for i,tank in enumerate(TANK_IDS,1):
        volume = station.GetPrimAtPath('/TankStation/'+tank).GetAttribute('fluidmix:volumeMl').Get(1)
        if volume:
            instances[100:200,i*100:i*100+50] = i
        tanks[tank] = dict(instance_id=i,volume_ml=volume,pixels=5000 if volume else 0,
                           bbox_xyxy=[i*100,100,i*100+50,200] if volume else None)
        volumes['/TankStation/'+tank] = volume
    Image.fromarray(instances).save(folder/"0001.00.instances.png")
    Image.fromarray((instances>0).astype('uint8')*255).save(folder/"0001.00.liquid.png")
    summary = dict(stage_sha256=hashlib.sha256(Path(station.GetRootLayer().realPath).read_bytes()).hexdigest(),
                   records=[dict(frame=1,sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                 tanks=tanks,volumes_ml=volumes,liquid_pixels=15000)])
    (folder/'summary.json').write_text(json.dumps(summary))
    assert check_capture(folder,station.GetRootLayer().realPath)['passed']
    tanks['T100']['volume_ml'] += 1
    (folder/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='volume label'):
        check_capture(folder,station.GetRootLayer().realPath)
    tanks['T100']['volume_ml'] -= 1
    (folder/'summary.json').write_text(json.dumps(summary))
    instances[0,0] = 3
    Image.fromarray(instances).save(folder/"0001.00.instances.png")
    with pytest.raises(ValueError,match='inactive T300'):
        check_capture(folder,station.GetRootLayer().realPath)


def test_semantic_ids_handle_runtime_capitalization():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from render_tank_station import semantic_ids
    labels = {'6':{'class':'liquid_t100'},'7':{'class':'liquid_T200'},
              '8':{'class':'liquid_other'},'9':{'class':'BACKGROUND'}}
    assert semantic_ids(labels,'liquid_T100') == [6]
    assert semantic_ids(labels,'liquid_T200') == [7]
    assert semantic_ids(labels,'liquid_T300') == []
