# Progress Log

## Session 1
- Read `todo.md` and initialized planning files.
- Created `rescore_sim.py` to evaluate the Oracle ceiling of disentangling position from confidence.
- Ran the simulation. The Oracle upper bound for mR+@5 on Idea 1 DualTrack is 14.06%, compared to the baseline 3.12%.
- The hypothesis is confirmed: "子弹确实在膛里" (the queries successfully localized the secondary targets), but they are getting penalized by the confidence score.
- Formulated the next steps: to extract real GT-free signals (e.g. boundary regression sharpness) from the inference code to rescore the queries.

## Session 2 (Current)
- Attempted to implement GT-free rescoring using query-text cosine similarity. The resulting `mR+@5` was 1.48%.
- Attempted to implement GT-free rescoring using the average `saliency_scores` within the predicted time window. The resulting `mR+@5` was 1.48%.
- Both simple approaches failed to match the Oracle performance (14.06%), because these intermediate features are not directly trained with contrastive/localization losses that would align them with confidence. 
- Awaiting user's promised details on the specific "真实 GT-free 重打分的推理改造点".
