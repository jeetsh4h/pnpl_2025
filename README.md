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
- [ ] Feature engineer the data
    - [ ] mean and p95 of the modulus of the data (per sample, per channel)
    - [ ] mean and p95 of the product of the modulus of a sample for pairwise channels
    - [ ] save the feature engineered data within a human-readable file (csv)
