# The service owner's item selection, ported verbatim from the committed active list in
# Urgendalai/gep_livestock_provision (fao_data_livestock.py, livestock_value_codes): 65 FAO item
# codes including the milk items the name list below lacks. Code-based filtering is also robust
# to FAO's item-name revisions. PROVISIONAL until the owner confirms the list is current; his
# script filters value on Element Code 152 where this module uses 57 -- an open method question
# recorded in the tracker, not silently changed here.
OWNER_LIVESTOCK_VALUE_CODES = [
    1012, 1017, 1018, 1019, 1020, 1025, 1032, 1035, 1036, 1037,
    1044, 1055, 1058, 1062, 1069, 1070, 1073, 1077, 1080, 1084,
    1087, 1089, 1091, 1094, 1097, 1098, 1108, 1111, 1120, 1122,
    1124, 1127, 1128, 1129, 1130, 1137, 1141, 1144, 1151, 1154,
    1158, 1161, 1163, 1166, 1176, 1182, 1183, 1185,
    2044, 867, 868, 869, 882, 944, 947, 948, 949, 951, 972, 977,
    978, 979, 982, 987, 995,
]

DEFAULT_LIVESTOCK_ITEMS = [
    "Meat of asses, fresh or chilled (indigenous)",
    "Meat of buffalo, fresh or chilled (indigenous)",
    "Meat of camels, fresh or chilled (indigenous)",
    "Meat of chickens, fresh or chilled (indigenous)",
    "Meat of ducks, fresh or chilled (indigenous)",
    "Meat of geese, fresh or chilled (indigenous)",
    "Meat of goat, fresh or chilled (indigenous)",
    "Meat of mules, fresh or chilled (indigenous)",
    "Meat of other domestic camelids, fresh or chilled (indigenous)",
    "Meat of pig with the bone, fresh or chilled (indigenous)",
    "Meat of pigeons and other birds n.e.c., fresh, chilled or frozen (indigenous)",
    "Meat of rabbits and hares, fresh or chilled (indigenous)",
    "Meat of sheep, fresh or chilled (indigenous)",
    "Meat of turkeys, fresh or chilled (indigenous)",
    "Horse meat, fresh or chilled (indigenous)",
    "Hen eggs in shell, fresh",
    "Eggs from other birds in shell, fresh, n.e.c.",
    "Game meat, fresh, chilled or frozen",
    "Meat of cattle with the bone, fresh or chilled",
    "Meat of chickens, fresh or chilled",
    "Meat of goat, fresh or chilled",
    "Meat of pig with the bone, fresh or chilled",
    "Meat of sheep, fresh or chilled",
    "Meat of turkeys, fresh or chilled",
    "Meat of camels, fresh or chilled",
    "Horse meat, fresh or chilled",
    "Meat of rabbits and hares, fresh or chilled",
    "Natural honey",
    "Other meat n.e.c. (excluding mammals), fresh, chilled or frozen",
]
