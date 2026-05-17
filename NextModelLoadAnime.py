import os
import json
import numpy as np
import open3d as o3d
import colorsys
initial_camera = None

# ============================================================
# Joint
# ============================================================

class Joint:
    def __init__(
        self,
        joint_type,
        axis,
        pivot=None,
        path=None,
        axisno=None
    ):

        self.type = joint_type

        self.axis = np.array(
            axis,
            dtype=float
        )

        n = np.linalg.norm(self.axis)

        if n > 0:
            self.axis /= n

        self.pivot = np.array(
            pivot if pivot is not None else [0, 0, 0],
            dtype=float
        )

        self.path = path
        self.axisno = axisno


# ============================================================
# SceneNode
# ============================================================

class SceneNode:
    def __init__(self, name):
        self.name = name

        self.children = []

        self.meshes = []

        self.local_T = np.eye(4)
        self.world_T = np.eye(4)
        self.def_T = np.eye(4)

        self.joint = None

        self.joint_value = 0.0


# ============================================================
# Motion
# ============================================================

class MotionClip:
    def __init__(self, name, sequence):

        self.name = name

        self.sequence = sequence

        self.total_duration = sum(
            s["duration"]
            for s in sequence
        )

        # ----------------------------------------------------
        # bake cumulative
        # ----------------------------------------------------

        self.segments = []

        current_time = 0.0
        current_value = 0.0

        for seg in sequence:

            duration = seg["duration"]

            speed = seg["speed"]

            end_value = (
                current_value
                + speed * duration
            )

            self.segments.append({

                "start_time": current_time,
                "end_time": current_time + duration,

                "start_value": current_value,
                "end_value": end_value,

                "speed": speed
            })

            current_time += duration

            current_value = end_value

        self.total_value = current_value

class MotionBinding:
    def __init__(
        self,
        target,
        clip,
        time_offset=0.0
    ):

        self.target = target

        self.clip = clip

        self.time_offset = time_offset


# ============================================================
# transform
# ============================================================

def make_transform(transform_def):
    T = np.eye(4)

    if transform_def is None:
        return T

    # --------------------------------------------------------
    # position
    # --------------------------------------------------------

    pos = transform_def.get(
        "position",
        [0, 0, 0]
    )

    T[:3, 3] = pos

    # --------------------------------------------------------
    # scale
    # --------------------------------------------------------

    scale = transform_def.get(
        "scale",
        [1, 1, 1]
    )

    S = np.eye(4)

    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]

    # --------------------------------------------------------
    # rotation
    # --------------------------------------------------------

    rot = transform_def.get(
        "rotation",
        [0, 0, 0]
    )

    rx, ry, rz = np.radians(rot)

    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    return T @ R @ S


# ============================================================
# joint parser
# ============================================================

def parse_joint(joint_def):

    if joint_def is None:
        return None

    return Joint(
        joint_type=joint_def["type"],
        axis=joint_def["axis"],
        pivot=joint_def.get(
            "pivot",
            [0, 0, 0]
        ),
        path=joint_def.get("path"),
        axisno=joint_def.get("axisno")
    )

# ============================================================
# joint transform
# ============================================================

def make_joint_transform(
    joint,
    value
):

    if joint is None:
        return np.eye(4)

    # ========================================================
    # LINEAR
    # ========================================================

    if joint.type == "linear":

        T = np.eye(4)

        T[:3, 3] = (
            joint.axis * value
        )

        return T

    # ========================================================
    # ROTATE
    # ========================================================

    elif joint.type == "rotate":

        axis = joint.axis

        pivot = joint.pivot

        rad = np.radians(value)

        x, y, z = axis

        c = np.cos(rad)
        s = np.sin(rad)

        R = np.array([
            [
                c + x*x*(1-c),
                x*y*(1-c) - z*s,
                x*z*(1-c) + y*s,
                0
            ],
            [
                y*x*(1-c) + z*s,
                c + y*y*(1-c),
                y*z*(1-c) - x*s,
                0
            ],
            [
                z*x*(1-c) - y*s,
                z*y*(1-c) + x*s,
                c + z*z*(1-c),
                0
            ],
            [0, 0, 0, 1]
        ])

        T1 = np.eye(4)
        T1[:3, 3] = -pivot

        T2 = np.eye(4)
        T2[:3, 3] = pivot

        return T2 @ R @ T1

    return np.eye(4)


# ============================================================
# build node
# ============================================================

# ============================================================
# Array Transform
# ============================================================
def rotation_from_direction(direction, up=np.array([0, 1, 0])):

    z = direction / np.linalg.norm(direction)

    x = np.cross(up, z)

    if np.linalg.norm(x) < 1e-6:
        up = np.array([1, 0, 0])
        x = np.cross(up, z)

    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    R = np.eye(4)

    R[:3, 0] = x
    R[:3, 1] = y
    R[:3, 2] = z

    return R

def compute_array_transform(defn, i):

    mode = defn["mode"]
    align = defn.get("align_to_path", False)

    # ========================================================
    # LINE
    # ========================================================
    if mode == "line":

        direction = np.array(
            defn["direction"],
            dtype=float
        )

        direction /= np.linalg.norm(direction)

        start = np.array(
            defn.get("start", [0, 0, 0]),
            dtype=float
        )

        spacing = defn["spacing"]

        pos = start + direction * spacing * i

        T = np.eye(4)
        T[:3, 3] = pos

        if align:

            R = rotation_from_direction(
                direction
            )

            return T @ R

        return T

    # ========================================================
    # ARC
    # ========================================================
    elif mode == "arc":
        center = np.array(
            defn["center"],
            dtype=float
        )
        radius = defn["radius"]
        axis = np.array(
            defn["axis"],
            dtype=float
        )
        axis /= np.linalg.norm(axis)

        a0 = defn["start_angle"]
        a1 = defn["end_angle"]

        t = i / max(defn["count"] - 1, 1)
        angle = a0 + (a1 - a0) * t
        rad = np.radians(angle)

        # Rodrigues rotation
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R3 = (
            np.eye(3)
            + np.sin(rad) * K
            + (1 - np.cos(rad)) * (K @ K)
        )

        # base vector
        base = np.array([radius, 0, 0])
        pos = R3 @ base

        T = np.eye(4)
        T[:3, 3] = center + pos

        if align:
            tangent = np.cross(axis, pos)
            if np.linalg.norm(tangent) > 0:
                tangent /= np.linalg.norm(tangent)

            R = rotation_from_direction(
                tangent
            )
            return T @ R

        return T

    else:
        raise ValueError(
            f"Unknown array mode: {mode}"
        )
    
def expand_array(
    defn,
    defs,
    base_dir,
    path
):
    nodes = []
    count = defn["count"]
    ref = defn["ref"]

    for i in range(count):
        # --------------------------------
        # 子ノード生成
        # --------------------------------
        child_node = {
            "ref": ref
        }

        node = build_node(
            child_node,
            defs,
            base_dir,
            f"{path}_arr{i}"
        )

        # --------------------------------
        # array transform
        # --------------------------------
        array_T = compute_array_transform(
            defn,
            i
        )
        
        node.local_T = array_T @ make_transform(defn.get("transform"))

        # --------------------------------
        # 初期姿勢保存
        # --------------------------------
        node.initial_local_T = (
            node.local_T.copy()
        )

        nodes.append(node)

    return nodes

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    return mesh

def build_node(
    node_def,
    defs,
    base_dir,
    path=""
):

    ref = node_def["ref"]

    defn = defs[ref]

    current_path = (
        f"{path}.{ref}"
        if path
        else ref
    )

    node = SceneNode(current_path)
    node.local_T = make_transform(node_def.get("transform"))

    # ========================================================
    # mesh
    # ========================================================

    if defn["type"] == "mesh":
        mesh_path = os.path.join(
            base_dir,
            defn["file"]
        )
        mesh = o3d.io.read_triangle_mesh(
            mesh_path
        )
        mesh.compute_vertex_normals()
        node.meshes.append(mesh)
        
        node.def_T = make_transform(defn.get("transform"))

    # ========================================================
    # node
    # ========================================================

    elif defn["type"] == "node":
        node.joint = parse_joint(
            defn.get("joint")
        )

        for child_def in defn.get(
            "children",
            []
        ):
            child = build_node(
                child_def,
                defs,
                base_dir,
                current_path
            )
            node.children.append(child)

    # --------------------------------------------------------
    # array
    # --------------------------------------------------------
    elif defn["type"] == "array":
        children = expand_array(defn, defs, base_dir, current_path)
        node.children.extend(children)

    return node


# ============================================================
# update world transform
# ============================================================

def update_world_transform(
    node,
    parent_T=np.eye(4)
):

    joint_T = make_joint_transform(
        node.joint,
        node.joint_value
    )

    node.world_T = (
        parent_T
        @ node.local_T
        @ joint_T
    )

    for child in node.children:

        update_world_transform(
            child,
            node.world_T
        )


# ============================================================
# motion loader
# ============================================================

def load_motion_file(path):

    with open(path, "r") as f:

        data = json.load(f)

    # --------------------------------------------------------
    # motions
    # --------------------------------------------------------

    clips = {}

    for name, motion_def in data[
        "motions"
    ].items():

        clips[name] = MotionClip(
            name,
            motion_def["sequence"]
        )

    # --------------------------------------------------------
    # bindings
    # --------------------------------------------------------

    bindings = []

    for b in data["bindings"]:

        bindings.append(
            MotionBinding(
                target=b["target"],
                clip=clips[b["motion"]],
                time_offset=b.get(
                    "time_offset",
                    0.0
                )
            )
        )

    return bindings


# ============================================================
# find node
# ============================================================

def find_node(node, path):

    if path in node.name:
        return node

    for child in node.children:

        result = find_node(
            child,
            path
        )

        if result is not None:
            return result

    return None


# ============================================================
# evaluate clip
# ============================================================
def evaluate_clip(
    clip,
    t
):

    if clip.total_duration <= 0:
        return 0.0

    # --------------------------------------------------------
    # loop
    # --------------------------------------------------------

    t = t % clip.total_duration

    for seg in clip.segments:

        if seg["start_time"] <= t < seg["end_time"]:

            local_t = (
                t - seg["start_time"]
            )

            value = (
                seg["start_value"]
                + seg["speed"] * local_t
            )

            return value

    return 0.0

# ============================================================
# apply motions
# ============================================================

def apply_motion_bindings(
    roots,
    bindings,
    t
):

    for binding in bindings:

        for root in roots:

            node = find_node(
                root,
                binding.target
            )

            if node is not None:

                value = evaluate_clip(
                    binding.clip,
                    t + binding.time_offset
                )

                node.joint_value = value

                break


# ============================================================
# collect meshes
# ============================================================

COLOR_LIST = [
    [0.7, 0.7, 0.7],  # light gray
              ]

for i in range(10):

    h = i / 10      # 色相
    s = 0.4        # 彩度低め
    v = 0.9        # 明るめ

    rgb = colorsys.hsv_to_rgb(h, s, v)

    COLOR_LIST.append(rgb)

def collect_meshes(
    node,
    out_list,
    color_index=0
):

    # --------------------------------------------------------
    # JointがあるNodeを通過したら次の色へ
    # --------------------------------------------------------

    if node.joint is not None:
        color_index += 1

    color = COLOR_LIST[
        color_index % len(COLOR_LIST)
    ]

    for mesh in node.meshes:

        mesh.paint_uniform_color(color)

        out_list.append(
            (mesh, node.world_T @ node.def_T)
        )

    for child in node.children:

        collect_meshes(
            child,
            out_list,
            color_index
        )

# ============================================================
# export merged STL
# ============================================================

def export_scene_as_stl(
    roots,
    output_path="export.stl"
):

    merged = o3d.geometry.TriangleMesh()

    all_meshes = []

    for root in roots:
        collect_meshes(
            root,
            all_meshes
        )

    for mesh, world_T in all_meshes:

        # ----------------------------------------
        # 元メッシュを壊さないようコピー
        # ----------------------------------------

        m = o3d.geometry.TriangleMesh(mesh)

        # ----------------------------------------
        # merge
        # ----------------------------------------

        merged += m

    # --------------------------------------------
    # 法線再計算
    # --------------------------------------------

    merged.compute_vertex_normals()

    # --------------------------------------------
    # STL出力
    # --------------------------------------------

    o3d.io.write_triangle_mesh(
        output_path,
        merged
    )

    print(f"[EXPORT] {output_path}")

# ============================================================
# S key export
# ============================================================

def on_key_s(vis):
    export_scene_as_stl(roots, "scene_export.stl")
    return False

# ============================================================
# R key camera reset
# ============================================================

def on_key_r(vis):
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        initial_camera,
        allow_arbitrary=True
    )
    return False

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    MODEL_JSON = "WPSmodel.json"
    MOTION_JSON = "WPSmotion.json"

    DT = 1 / 60

    # ============================================================
    # load model
    # ============================================================

    with open(MODEL_JSON, "r") as f:
        model_data = json.load(f)

    defs = model_data["definitions"]
    base_dir = os.path.dirname(
        MODEL_JSON
    )

    roots = []

    for scene_node in model_data["scene"]:
        root = build_node(
            scene_node,
            defs,
            base_dir
        )
        roots.append(root)

    # ============================================================
    # load motions
    # ============================================================

    bindings = load_motion_file(
        MOTION_JSON
    )

    # ============================================================
    # build mesh list
    # ============================================================
    for root in roots:
        update_world_transform(root)

    all_meshes = []

    for root in roots:
        collect_meshes(
            root,
            all_meshes
        )

    geometry_list = []

    for mesh, world_T in all_meshes:
        mesh.transform(world_T)
        geometry_list.append(mesh)


    # ============================================================
    # visualizer
    # ============================================================

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord("S"), on_key_s)
    vis.register_key_callback(ord("R"), on_key_r)
    vis.create_window()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(axis)

    for g in geometry_list:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([0,0,0])
    ctr = vis.get_view_control()
    ctr.set_front([1, 1, 1])
    initial_camera = ctr.convert_to_pinhole_camera_parameters()

    # ============================================================
    # animation loop
    # ============================================================

    t = 0.0

    while True:
        
        if not vis.poll_events():
            break

        # --------------------------------------------------------
        # reset previous transform
        # --------------------------------------------------------

        for mesh, world_T in all_meshes:
            mesh.transform(
                np.linalg.inv(world_T)
            )

        # --------------------------------------------------------
        # motion update
        # --------------------------------------------------------

        apply_motion_bindings(
            roots,
            bindings,
            t
        )

        # --------------------------------------------------------
        # transform update
        # --------------------------------------------------------

        for root in roots:
            update_world_transform(root)

        # --------------------------------------------------------
        # apply transform
        # --------------------------------------------------------

        all_meshes.clear()

        for root in roots:
            collect_meshes(
                root,
                all_meshes
            )

        for mesh, world_T in all_meshes:
            mesh.transform(world_T)

        # --------------------------------------------------------
        # redraw
        # --------------------------------------------------------

        for g in geometry_list:
            vis.update_geometry(g)

        vis.poll_events()
        vis.update_renderer()

        # --------------------------------------------------------
        # time
        # --------------------------------------------------------

        t += DT