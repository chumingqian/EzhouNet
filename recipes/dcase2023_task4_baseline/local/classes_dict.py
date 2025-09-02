"""
we store here a dict where we define the encodings for all classes in DESED task.
"""

from collections import OrderedDict
#
# classes_labels = OrderedDict(
#     {
#         "Alarm_bell_ringing": 0,
#         "Blender": 1,
#         "Cat": 2,
#         "Dishes": 3,
#         "Dog": 4,
#         "Electric_shaver_toothbrush": 5,
#         "Frying": 6,
#         "Running_water": 7,
#         "Speech": 8,
#         "Vacuum_cleaner": 9,
#     }
# )


classes_labels = OrderedDict(
    {
        "Normal": 0,
        "Rhonchi": 1,
        "Wheeze": 2,
        "Stridor": 3,
        "Coarse Crackle": 4,
        "Fine Crackle": 5,
        "Wheeze+Crackle": 6,
    }
)



classes_6labels = OrderedDict(
    {
        "Normal": 0,
        "Rhonchi": 1,
        "Wheeze": 2,
        "Stridor": 3,
        "Coarse Crackle": 4,
        "Fine Crackle": 5,
    }
)


classes_5labels = OrderedDict(
    {
        "Normal": 0,
        "Rhonchi": 1,
        "Wheeze": 2,
        "Stridor": 3,
        "Crackle": 4,
    }
)





binary_labels = OrderedDict(
    {
        "Normal": 0,
        "Abnormal": 1,
    }
)



# Normal, Rhonchi, Wheeze, Stridor, Coarse Crackle, Fine Crackle, or Wheeze+Crackle