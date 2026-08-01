"""Semantic objects available to the text agent in this scene."""

SCENE_OBJECTS = [
    {
        "name": "part_A",
        "category": "part_A",
        "type": "visual_category",
        "aliases": ["零件A"],
        "properties": {
            "graspable": True,
            "instances": "由head_camera检测结果在运行时动态生成",
        },
    },
    {
        "name": "part_B",
        "category": "part_B",
        "type": "visual_category",
        "aliases": ["零件B"],
        "properties": {
            "graspable": True,
            "instances": "由head_camera检测结果在运行时动态生成",
        },
    },
    {
        "name": "box",
        "category": "container",
        "type": "container",
        "aliases": ["盒子", "料盒", "容器", "塑料框"],
        "properties": {
            "placeable": True,
            "placement_policy": "根据RGB-D可见轮廓和当前占用动态选择空闲区域",
        },
    },
]


SCENE_CONVENTIONS = [
    "用户说“拿零件A”且未指定位置或数量时，表示将head_camera当前检测到的全部零件A放入box；实例数量不能预设。",
    "用户说“拿零件B”且未指定位置或数量时，表示将head_camera当前检测到的全部零件B放入box；实例数量不能预设。",
    "用户指定零件类别及左侧、中间或右侧时，只操作该类别对应位置的零件。",
    "用户只说“拿零件”而没有说明A或B时，必须请求用户澄清。",
    "无法确定目标时必须请求用户澄清。",
    "闭环每轮由Agent选择一个零件和一只机械臂，自主编排完整的pick、lift、place_in、retreat，并且只能引用本轮视觉和规划器提供的合法参数。",
]
