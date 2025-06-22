import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc
import random

from blenderproc.python.types.EntityUtility import Entity
from blenderproc.python.types.MaterialUtility import Material

from .utils.geometry import rand_rotation_euler
from .utils.material import group_and_filter_material_slots_by_cp

ALLOWED_ROTATION_MODES = [
    "XYZ",  # XYZ Euler.XYZ Rotation Order - prone to Gimbal Lock (default).
    "XZY",  # XZY Euler.XZY Rotation Order - prone to Gimbal Lock.
    "YXZ",  # YXZ Euler.YXZ Rotation Order - prone to Gimbal Lock.
    "YZX",  # YZX Euler.YZX Rotation Order - prone to Gimbal Lock.
    "ZXY",  # ZXY Euler.ZXY Rotation Order - prone to Gimbal Lock.
    "ZYX",  # ZYX Euler.ZYX Rotation Order - prone to Gimbal Lock.
]


def randomize_drone_geometry(
    frame: int,
    entities: list[Entity],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
):
    # the first entity is always the ancestor
    model = entities[0]
    assert model.blender_obj.rotation_mode in ALLOWED_ROTATION_MODES, (
        "The rotation mode should be in Euler."
    )

    # randomly sample an euler angle
    euler = rand_rotation_euler(x_range, y_range, z_range)
    model.set_rotation_euler(euler, frame=frame)


def randomize_drone_materials(
    entities: list[Entity],
    materials: list[Material],
    group_cp_name: str = "group_name",
    filter_cp_name: str = "material_randomization",
):
    # get material slots which require material_randomization
    material_slots_groups = group_and_filter_material_slots_by_cp(
        entities, group_cp_name=group_cp_name, filter_cp_name=filter_cp_name
    )

    # randomly apply a material for each group
    for material_slots in material_slots_groups.values():
        num_materials = len(materials)

        # Consider the original material in the material slot
        if any(
            mesh.blender_obj.material_slots[i].material is not None
            for mesh, i in material_slots
        ):
            assert all(
                mesh.blender_obj.material_slots[i].material is not None
                for mesh, i in material_slots
            ), "Every material slot in the same group should have a material."
            num_materials += 1

        material_i = random.randrange(0, num_materials)
        if material_i < len(materials):
            material = materials[material_i]
            for mesh, i in material_slots:
                mesh.set_material(i, material)


def randomize_drone_properties(
    frame: int,
    entities: list[Entity],
    materials: list[Material],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    group_cp_name: str = "group_name",
    filter_cp_name: str = "material_randomization",
):
    # sample an euler angle
    randomize_drone_geometry(
        frame, entities, x_range=x_range, y_range=y_range, z_range=z_range
    )

    # material randomization
    randomize_drone_materials(
        entities,
        materials,
        group_cp_name=group_cp_name,
        filter_cp_name=filter_cp_name,
    )
