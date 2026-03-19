# Future Work

## 1. Incorporate the Disturbed YouTube Dataset in Training

Expand the training pipeline to include the Disturbed YouTube dataset so the model can learn from a broader and more realistic set of unsafe and borderline content examples. This should improve generalization and make recommendations more robust across different video types.

## 2. Add Image Analysis of YouTube Thumbnails

Extend the current text-focused system into a multimodal pipeline by analyzing YouTube thumbnails alongside metadata and transcripts. Thumbnail analysis could help detect visual risk signals that are not visible in titles, descriptions, or spoken content.

## 3. Improve the Age-Aware Policy Decision Maker

Refine the policy layer so it better captures differences between age groups and handles ambiguous cases more consistently. Future versions could use more structured thresholds, risk weighting, and policy rules tailored to developmental sensitivity.

## 4. Improve the Explanation Component with XAI

Strengthen the explanation module by using explainable AI techniques that make model outputs easier to interpret and audit. This can help show which inputs most influenced a recommendation and improve trust in the system.

## 5. Evaluate Results More Deeply

Carry out a more comprehensive evaluation of model and policy performance across datasets, age groups, and recommendation categories. This should include quantitative metrics, error analysis, and comparisons between model predictions and final policy decisions.
