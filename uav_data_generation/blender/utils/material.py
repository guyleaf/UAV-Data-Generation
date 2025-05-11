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
    def get_cp_values(obj: MeshObject, cp_name: str, defaults):
        """
        Get a list of values from custom property
            1. if the cp_name is not defined, use the default value.
            2. if the cp_value is a single value, broadcast to every slot
            3. if the cp_value is a list, use it directly.
        """
        cp_values = get_cp(obj, cp_name, default=defaults)
        if isinstance(cp_values, idprop.types.IDPropertyArray):
            cp_values = cp_values.to_list()

        # check the length of cp_values == num_slots
        num_slots = max(len(obj.blender_obj.material_slots), 1)
        if isinstance(cp_values, list):
            assert len(cp_values) == num_slots
        else:
            cp_values = [cp_values] * num_slots
        return cp_values

    # the first mesh is always the ancestor of all UAV meshes
    uav_model = meshes[0]
    default_filter_cp_value = get_cp(
        uav_model, f"default_{filter_cp_name}", default=True
    )
    assert isinstance(default_filter_cp_value, bool)

    # grouping by group_cp_name & filtering by filter_cp_name
    groups = defaultdict(list)
    for mesh in meshes:
        name = mesh.get_name()
        num_slots = max(len(mesh.blender_obj.material_slots), 1)

        # grouping by group_cp_name
        default_value = [f"{name}_{i}" for i in range(num_slots)]
        group_cp_values: list[str] = get_cp_values(
            mesh, group_cp_name, defaults=default_value
        )
        assert all(isinstance(cp_value, str) for cp_value in group_cp_values)

        # filtering by filter_cp_name
        default_value = [default_filter_cp_value] * num_slots
        filter_cp_values: list[bool] = get_cp_values(
            mesh, filter_cp_name, defaults=default_value
        )
        assert all(isinstance(cp_value, bool) for cp_value in filter_cp_values)

        for i, group_cp in filter(
            lambda item: filter_cp_values[item[0]], enumerate(group_cp_values)
        ):
            groups[group_cp].append((mesh, i))
    return groups
