import pickle
import os, sys
sys.path.append('../')
sys.path.append(os.getcwd())

result_location = 'record/Middle_RandomSeed_0_CurvePoint_0.05_Ratio_0.05_FixStart_True_FixEnd_True_Epoch_1_TSPCEpoch_3.pkl'
with open(result_location, 'rb') as file:
    (first_stage_curve_acc_result, second_stage_curve_acc_result,
     first_stage_curve_asr_result, second_stage_curve_asr_result,
     first_stage_asr_result, second_stage_asr_result,
     first_stage_acc_result, second_stage_acc_result) = pickle.load(file)
    print(file)