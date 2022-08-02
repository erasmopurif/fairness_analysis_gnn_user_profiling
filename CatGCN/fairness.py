import numpy as np
import pandas as pd

class Fairness(object):
    """
    Compute fairness metrics
    """
    def __init__(self, df_profile, test_nodes_idx, targets, predictions, sens_attr, neptune_run):
        self.sens_attr = sens_attr
        self.neptune_run = neptune_run
        self.neptune_run["sens_attr"] = self.sens_attr
        self.df_profile = df_profile
        self.test_nodes_idx = test_nodes_idx.cpu().detach().numpy()
        self.true_y = targets # target variables
        self.pred_y = predictions # prediction of the classifier
        self.sens_attr_array = self.df_profile[self.sens_attr].values # sensitive attribute values
        self.sens_attr_values = self.sens_attr_array[self.test_nodes_idx]
        self.s0 = self.sens_attr_values == 0
        self.s1 = self.sens_attr_values == 1
        self.y1_s0 = np.bitwise_and(self.true_y==1, self.s0)
        self.y1_s1 = np.bitwise_and(self.true_y==1, self.s1)
        self.y0_s0 = np.bitwise_and(self.true_y==0, self.s0)
        self.y0_s1 = np.bitwise_and(self.true_y==0, self.s1)

    
    def statistical_parity(self):
        ''' P(y^=1|s=0) = P(y^=1|s=1) '''
        # stat_parity = abs(sum(self.pred_y[self.s0]) / sum(self.s0) - sum(self.pred_y[self.s1]) / sum(self.s1))
        stat_parity_s0 = sum(self.pred_y[self.s0]) / sum(self.s0)
        stat_parity_s1 = sum(self.pred_y[self.s1]) / sum(self.s1)
        stat_parity_diff = stat_parity_s0 - stat_parity_s1
        self.neptune_run["fairness/SP_s0"] = stat_parity_s0
        self.neptune_run["fairness/SP_s1"] = stat_parity_s1
        self.neptune_run["fairness/SPD"] = stat_parity_diff
        print(" Statistical Parity Difference (SPD): {:.4f}".format(stat_parity_diff))

    
    def equal_opportunity(self):
        ''' P(y^=1|y=1,s=0) = P(y^=1|y=1,s=1) '''
        # equal_opp = abs(sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0) - sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1))
        equal_opp_s0 = sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0)
        equal_opp_s1 = sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1)
        equal_opp_diff = equal_opp_s0 - equal_opp_s1
        self.neptune_run["fairness/EO_s0"] = equal_opp_s0
        self.neptune_run["fairness/EO_s1"] = equal_opp_s1
        self.neptune_run["fairness/EOD"] = equal_opp_diff
        print(" Equal Opportunity Difference (EOD): {:.4f}".format(equal_opp_diff))


    def overall_accuracy_equality(self):
        ''' P(y^=0|y=0,s=0) + P(y^=1|y=1,s=0) = P(y^=0|y=0,s=1) + P(y^=1|y=1,s=1) '''
        oae_s0 = np.count_nonzero(self.pred_y[self.y0_s0]==0) / sum(self.y0_s0) + sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0)
        oae_s1 = np.count_nonzero(self.pred_y[self.y0_s1]==0) / sum(self.y0_s1) + sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1)
        # oae_diff = abs(oae_s0 - oae_s1)
        oae_diff = oae_s0 - oae_s1
        self.neptune_run["fairness/OAE_s0"] = oae_s0
        self.neptune_run["fairness/OAE_s1"] = oae_s1
        self.neptune_run["fairness/OAED"] = oae_diff
        print(" Overall Accuracy Equality Difference (OAED): {:.4f}".format(oae_diff))


    def treatment_equality(self):
        ''' P(y^=1|y=0,s=0) / P(y^=0|y=1,s=0) = P(y^=1|y=0,s=1) / P(y^=0|y=1,s=1) '''
        te1_s0 = (sum(self.pred_y[self.y0_s0]) / sum(self.y0_s0)) / (np.count_nonzero(self.pred_y[self.y1_s0]==0) / sum(self.y1_s0))
        te1_s1 = (sum(self.pred_y[self.y0_s1]) / sum(self.y0_s1)) / (np.count_nonzero(self.pred_y[self.y1_s1]==0) / sum(self.y1_s1))
        te_diff_1 = te1_s0 - te1_s1
        abs_ted_1 = abs(te_diff_1)
        ''' P(y^=0|y=1,s=0) / P(y^=1|y=0,s=0) = P(y^=0|y=1,s=1) / P(y^=1|y=0,s=1) '''
        te0_s0 = (np.count_nonzero(self.pred_y[self.y1_s0]==0) / sum(self.y1_s0)) / (sum(self.pred_y[self.y0_s0]) / sum(self.y0_s0))
        te0_s1 = (np.count_nonzero(self.pred_y[self.y1_s1]==0) / sum(self.y1_s1)) / (sum(self.pred_y[self.y0_s1]) / sum(self.y0_s1))
        te_diff_0 = te0_s0 - te0_s1
        abs_ted_0 = abs(te_diff_0)

        # te_diff = min(te_diff_1, te_diff_0)
        if abs_ted_0 < abs_ted_1:
            te_s0 = te0_s0
            te_s1 = te0_s1
            te_diff = te_diff_0
        else:
            te_s0 = te1_s0
            te_s1 = te1_s1
            te_diff = te_diff_1

        self.neptune_run["fairness/TE_s0"] = te_s0
        self.neptune_run["fairness/TE_s1"] = te_s1
        self.neptune_run["fairness/TED"] = te_diff
        print(" Treatment Equality Difference (TED): {:.4f}".format(te_diff))
