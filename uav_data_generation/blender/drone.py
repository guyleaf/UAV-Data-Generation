import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import random

from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject

from .utils.geometry import rand_rotation_euler
from .utils.material import group_and_filter_material_slots_by_cp


def randomize_drone_properties(
    uav_components: list[MeshObject],
    materials: list[Material],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    group_cp_name: str = "group_name",
    filter_cp_name: str = "material_randomization",
) -> int:
    # the first component is always the ancestor of all UAV components
    uav_model = uav_components[0]

    # get material slots which require material_randomization
    material_slots_groups = group_and_filter_material_slots_by_cp(
        uav_components, group_cp_name=group_cp_name, filter_cp_name=filter_cp_name
    )

    # 1. randomly sample a frame for animation
    frame = random.randint(0, 249)

    # 2. randomly sample an euler angle
    euler = rand_rotation_euler(x_range, y_range, z_range)
    uav_model.set_rotation_euler(euler, frame=frame)

    # 3. randomly apply a material for each group
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

    return frame
