# 2024-04-11

## Notes

- Alex did the EDA (Exploratory Data Analysis)
- Possible Model: **Gradient-boosted tree model**
- For the Assignment Question "If I have a budget of €100,000, what kind of houses will I be able to buy?":
  - Should we derive this directly using our prediction model, or should we do an explicit analysis?
- Data Preprocessing is missing

---

# 2024-04-14

## Agenda

- [ ] Goal: **Task allocation from EDA across the team**
  - [ ] Section "Key findings" 
  - [ ] Section "Further Investigations"
- [ ] Weekly Meeting scheduled
- [ ] Deadline for the assigned Tasks set

## Notes

- 2024-05-17_14-43-27 log directory contains the best results using CV, 
  - Best CV RMSE: 19.700122942786038
  - Test RMSE: 18.802622435416964
  - Test MSE: 353.53861044884536
  - Test MAE: 14.030347668707856
  - Test R^2: 0.9648192382792768
- 2024-05-17_15-00-26 log directory contains the best results without CV, 
  - Test RMSE: 17.821948483946873
  - Test MSE: 317.6218477644562
  - Test MAE: 13.108861363310897
  - Test R^2: 0.9683933290077974

Models without CV seem to have better results compared to CV on the test set (maybe because we use more data to train? and we just have 941 samples)

Experiment 2024-05-22_15-49-02 has been trained on the data without categorical features ranking and has an RMSE of 16.63 on the test set.

Experiment 2024-05-26_13-41-44 contains the best results after grid search
Elapsed time for the Grid Search: 00:39:22.344
Best params: {'depth': 8, 'od_wait': 100, 'learning_rate': 0.1, 'iterations': 1000}