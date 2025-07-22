# The 2025 PNPL Competition
## Speech Detection and Phoneme Classification in the LibriBrain Dataset

Read about the competition [here](https://eval.ai/web/challenges/challenge-page/2504/phases).

Currently working on  
**Phase: Speech Detection (Standard)**

Train a model to distinguish speech vs. silence based on brain activity measured by MEG during a listening session.

> MEG is magnetoencephalography

## Ends on: Aug 1, 2025 5:29:59 PM IST

# TODO

- [x] Save the data in /mnt/cai-data and then make a symlink to it
- [x] Download *all* the data
- [x] Train the default model given in the notebook
- [ ] Perform EDA to check differences between both labels
  - [ ] Check mean, min, max, p95 of channels between the two labels
  - [ ] Make a histogram of it
  - [ ] Save it in a CSV
- [ ] Feature engineer the data
  - [ ] mean and p95 of the modulus of the data (per sample, per channel)
  - [ ] correlation coefficient between channels
  - [ ] save the feature engineered data within a human-readable file (csv)
- [ ] Train a XGBoost model on this data
