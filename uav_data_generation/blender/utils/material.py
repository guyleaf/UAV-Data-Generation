from typing import Any, Sequence, Union

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


def get_cp_list_for_material_slots(
    obj: MeshObject, cp_name: str, defaults: Union[Any, Sequence[Any]]
):
    """
    Get a list of values from custom property
        1. if the cp_name is not defined, use the default value.
        2. if the cp_value is a single value, broadcast to every slot
        3. if the cp_value is a list, use it directly.
    """
    num_slots = len(obj.blender_obj.material_slots)
    if not isinstance(defaults, Sequence):
        defaults = [defaults] * num_slots

    cp_list = get_cp(obj, cp_name, default=defaults)
    if isinstance(cp_list, idprop.types.IDPropertyArray):
        cp_list = cp_list.to_list()
    # workaround, support array of strings by using ',' delimiter
    elif isinstance(cp_list, str):
        cp_list = cp_list.split(",")
        if len(cp_list) == 1:
            cp_list = cp_list[0]
        else:
            cp_list = [
                cp_value or default for cp_value, default in zip(cp_list, defaults)
            ]

    # check the length of cp_values == num_slots
    if isinstance(cp_list, Sequence):
        assert len(cp_list) == num_slots
    else:
        cp_list = [cp_list] * num_slots
    return list(cp_list)


def group_and_filter_material_slots_by_cp(
    meshes: list[MeshObject],
    group_cp_name: str = "group_name",
    filter_cp_name: str = "material_randomization",
) -> dict[str, list[tuple[MeshObject, int]]]:
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
        num_slots = len(mesh.blender_obj.material_slots)

        # grouping by group_cp_name
        default_values = [f"{name}_{i}" for i in range(num_slots)]
        group_cp_list: list[str] = get_cp_list_for_material_slots(
            mesh, group_cp_name, defaults=default_values
        )
        assert all(
            isinstance(cp_value, str) and len(cp_value) > 0
            for cp_value in group_cp_list
        )

        # filtering by filter_cp_name
        filter_cp_list: list[bool] = get_cp_list_for_material_slots(
            mesh, filter_cp_name, defaults=default_filter_cp_value
        )
        assert all(isinstance(cp_value, bool) for cp_value in filter_cp_list)

        for i, (group_cp, filter_cp) in enumerate(zip(group_cp_list, filter_cp_list)):
            if filter_cp:
                groups[group_cp].append((mesh, i))
    return groups
