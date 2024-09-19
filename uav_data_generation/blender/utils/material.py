import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

from collections import defaultdict
from operator import methodcaller

import idprop
from blenderproc.python.types.MeshObjectUtility import MeshObject

from .utils import get_cp


def collect_materials_by_cp(
    cp_name: str = "random_material", cp_value: bool = True
) -> list[bproc.types.Material]:
    materials = bproc.material.collect_all()
    materials = bproc.filter.by_cp(
        materials, cp_name, cp_value, filtered_data_type=bproc.types.Material
    )
    materials.sort(key=methodcaller("get_name"))
    print(f"Find {len(materials)} materials")
    return materials


def group_and_filter_material_slots_by_cp(
    meshes: list[MeshObject],
    group_cp_name: str = "group_name",
    filter_cp_name: str = "material_randomization",
) -> dict[str, list[tuple[MeshObject, int]]]:
    def get_cp_(obj: MeshObject, cp_name: str, default):
        cp_values = get_cp(obj, cp_name, default=default)
        if isinstance(cp_values, idprop.types.IDPropertyArray):
            cp_values = cp_values.to_list()

        # check the length of cp_values == num_slots
        num_slots = max(len(obj.blender_obj.material_slots), 1)
        if isinstance(cp_values, list):
            assert len(cp_values) == num_slots
        else:
            cp_values = [cp_values] * num_slots
        return cp_values

    groups = defaultdict(list)
    for mesh in meshes:
        name = mesh.get_name()
        num_slots = max(len(mesh.blender_obj.material_slots), 1)

        # grouping by group_cp_name
        # 1. if the cp is not defined, make every slot as an individual group
        # 2. if the cp_value is a single value, broadcast to every slot
        # 3. if the cp_value is a list, use it directly.
        default_value = [f"{name}_{i}" for i in range(num_slots)]
        group_cp_values: list[str] = get_cp_(mesh, group_cp_name, default=default_value)

        # filtering by filter_cp_name
        # 1. if the cp is not defined, make every slot require randomization
        # 2. if the cp_value is a single value, broadcast to every slot
        # 3. if the cp_value is a list, use it directly.
        default_value = [True] * num_slots
        filter_cp_values: list[bool] = get_cp_(
            mesh, filter_cp_name, default=default_value
        )

        for i, group_cp in filter(
            lambda item: filter_cp_values[item[0]], enumerate(group_cp_values)
        ):
            groups[group_cp].append((mesh, i))
    return groups
