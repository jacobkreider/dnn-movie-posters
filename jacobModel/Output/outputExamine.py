# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

performance = pd.DataFrame([[1, 2.955, .4069, 2.9779, .4008]
                            , [2, 2.893, .4148, 2.954, .4128]
                            , [3, 2.8605, .4362, 2.9225, .5021]
                            , [4, 2.8317, .4630, 2.9149, .4938]
                            , [5, 2.8096, .4813, 2.8826, .5311]
                            , [6, 2.7887, .4900, 2.8683, .5430]
                            , [7, 2.7695, .4984, 2.8702, .5453]
                            , [8, 2.7557, .4997, 2.8359, .4906]
                            , [9, 2.7341, .5080, 2.9608, .5324]
                            , [10, 2.7223, .5065, 2.8380, .5444]
                            , [11, 2.7604, .5119, 2.8193, .4781]
                            , [12, 2.6824, .5130, 2.8114, .5278]
                            , [13, 2.6717, .5131, 2.8246, .5255]
                            , [14, 2.6554, .5260, 2.8065, .5545]
                            , [15, 2.6358, .5243, 2.9666, .4772]
                            , [16, 2.6183, .5298, 2.8169, .5366]
                            , [17, 2.6025, .5328, 2.7990, .5099]
                            , [18, 2.5825, .5299, 2.8625, .5131]
                            , [19, 2.5701, .5365, 2.8241, .5177]
                            , [20, 2.5470, .5402, 2.8126, .5122]
                            , [21, 2.5284, .5369, 2.8767, .4892]
                            , [22, 2.5114, .5438, 2.8780, .5062]
                            , [23, 2.4841, .5439, 2.8964, .4942]
                            , [24, 2.4647, .5447, 2.8932, .5039]
                            , [25, 2.4441, .5473, 2.8932, .5039]
                            , [26, 2.4258, .5486, 2.8780, .5117]
                            , [27, 2.4089, .5559, 2.9264, .5071]
                            , [28, 2.3909, .5496, 2.9636, .4850]
                            , [29, 2.3705, .5546, 2.8715, .4961]
                            , [30, 2.3489, .5536, 2.9397, .5067]
                            , [31, 2.3294, .5599, 2.9926, .5182]
                            , [32, 2.3003, .5622, 2.9034, .5062]
                            , [33, 2.2889, .5658, 3.1551, .5430]
                            , [34, 2.2718, .5663, 2.9576, .4804]
                            , [35, 2.2492, .5628, 3.0461, .5090]
                            , [36, 2.2394, .5641, 3.0811, .4726]
                            , [37, 2.2099, .5747, 3.3163, .5274]
                            , [38, 2.2110, .5708, 3.3435, .5104]
                            , [39, 2.1953, .5707, 3.0651, .5131]
                            , [40, 2.1810, .5771, 3.1508, .5173]
                            , [41, 2.1661, .5718, 3.3805, .4827]
                            , [42, 2.1538, .5771, 3.4977, .5338]
                            , [43, 2.1408, .5767, 3.1636, .5122]
                            , [44, 2.1449, .5759, 3.0237, .4745]
                            , [45, 2.1304, .5809, 3.1382, .4979]
                            , [46, 2.1303, .5715, 3.3552, .5030]
                            , [47, 2.1153, .5838, 3.1931, .4988]
                            , [48, 2.1049, .5779, 3.1436, .4970]
                            , [49, 2.1016, .5810, 3.5098, .4975]
                            , [50, 2.0960, .5888, 3.1404, .5007]],
                            columns = ["Epoch", "Training Loss", "Training Accuracy"
                                       , "Validation Loss", "Validation Accuracy"])

import matplotlib.pyplot as plt

plt.plot(performance["Epoch"], performance["Training Loss"], 'b', label = "Training Loss")
plt.plot(performance["Epoch"], performance["Validation Loss"], '--b', label = "Validation Loss")
plt.show()

minVal = min(performance["Validation Loss"])

print(performance.loc[performance["Validation Loss"] == minVal] )

performance17epochs = pd.DataFrame([[1, 2.9570, .4067, 3.0246, .4008]
                                    , [2, 2.8915, .4111, 2.9661, .4040]
                                    , [3, 2.8584, .4288, 2.9532, .4745]
                                    , [4, 2.8324, .4648, 2.9859, .5288]
                                    , [5, 2.8146, .4802, 2.9460, .5117]
                                    , [6, 2.7921, .4897, 2.8854, .5223]
                                    , [7, 2.7688, .4979, 2.9374, .5430]
                                    , [8, 2.7573, .5017, 2.8493, .5209]
                                    , [9, 2.7378, .4995, 2.8211, .5108]
                                    , [10, 2.7229, .5050, 2.8160, .4731]
                                    , [11, 2.7051, .5106, 2.8330, .4942]
                                    , [12, 2.6900, .5168, 2.8219, .5283]
                                    , [13, 2.6728, .5153, 2.8126, .5159]
                                    , [14, 2.6578, .5220, 2.8029, .5389]
                                    , [15, 2.6379, .5234, 2.8290, .5223]
                                    , [16, 2.6196, .5247, 2.8010, .5163]
                                    , [17, 2.6053, .5269, 2.8330, .5007]],
                                    columns = )["Epoch", "Training Loss"
                                                , "Training Accuracy"
                                                , "Validation Loss"
                                                , "Validation Accuracy"])


