extra_nodes = {13056, 29698, 4868, 16388, 3078, 6919, 431112, 661001, 1291, 7181, 2065, 15378, 14356, 13844, 14612, 5912, 2846, 673824, 5153, 26658, 3617, 3108, 5412, 2086, 295, 1832, 28200, 3368, 5931, 18727, 2606, 28208, 561, 18743, 5945, 12346, 2875, 3641, 4159, 3135, 13889, 17986, 4420, 2373, 6215, 3402, 196685, 1363, 3925, 2904, 3418, 778075, 9818, 400989, 3678, 2911, 10849, 2404, 5220, 2407, 874, 5741, 5744, 175474, 2419, 115, 29562, 73857, 2694, 14215, 648, 1415, 2442, 74123, 438412, 8334, 10383, 1424, 4497, 6288, 57235, 1430, 49561, 195483, 8347, 16027, 7071, 364706, 2980, 420, 11176, 32424, 269994, 52906, 2732, 5804, 1198, 1967, 14000, 192693, 6074, 442, 14269, 2239, 3015, 5576, 11209, 5578, 7116, 73932, 7118, 3532, 7635, 15322, 16377, 3294, 25314, 8419, 8420, 15331, 2791, 2024, 1769, 194027, 16370, 16627, 3060, 757, 667127, 7672, 3065, 4606}
qa_nodes = {5765, 5926, 48710, 29705, 11503, 561, 2739, 7988, 2904, 4695, 1848, 16377, 10398}
ec_nodes = {99591, 222099, 225173, 737563, 31265, 31780, 231721, 66348, 306222, 7988, 48703, 252999, 232777, 232811, 31982, 31216, 63605, 222074, 222077, 222078}

print(len(extra_nodes), len(qa_nodes), len(ec_nodes))

print(extra_nodes.intersection(qa_nodes))
print(extra_nodes.intersection(ec_nodes))
print(ec_nodes.intersection(qa_nodes))


extra_nodes = extra_nodes - qa_nodes
ec_nodes = ec_nodes - qa_nodes
ec_nodes = ec_nodes - extra_nodes

print(extra_nodes.intersection(qa_nodes))
print(extra_nodes.intersection(ec_nodes))
print(ec_nodes.intersection(qa_nodes))

print(len(extra_nodes), len(qa_nodes), len(ec_nodes))
