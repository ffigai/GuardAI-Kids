# Future Work

## 1. Update the Training Dataset with Child-Specific Data

The most impactful improvement for a future version is updating the training dataset to include data designed specifically for child safety on video platforms. MetaHarm is a general-purpose harmful content dataset built for broad adult audiences — it was not designed around what is inappropriate *for children*, and this creates two structural gaps:

- **Missing child-appropriate negatives.** MetaHarm has no verified examples of child-safe content. Adding verified child-appropriate content as hard negatives would help the model learn that nursery rhyme vocabulary, animal nature content, and cartoon thumbnails are safe signals.
- **Label contamination from bad actors.** Harmful content that mimics children's vocabulary (adult parodies, inappropriate remixes using "nursery rhyme" or "kids songs" as cover) is labelled harmful in MetaHarm. The model learns these words as risk signals, which can penalise legitimate children's content.

The recommended dataset to add is the **Disturbed YouTube for Kids** dataset (Papadamou et al., ICWSM 2020), available at [https://zenodo.org/records/3632781](https://zenodo.org/records/3632781). It contains 4,797 manually annotated YouTube videos labelled as suitable, disturbing, restricted, or irrelevant — built specifically for detecting inappropriate content in the children's video space.

Practical angles for using this dataset:

- **Add "suitable" videos as hard negatives.** The "suitable" class provides verified child-appropriate content that is currently underrepresented in training. Adding these alongside MetaHarm harmful examples would improve precision on mainstream children's content.
- **Use as a held-out evaluation benchmark.** Evaluating the current model on the Disturbed YouTube for Kids dataset would quantify how well the model generalises to real-world children's YouTube content — a meaningful finding for the paper.
- **LLM-assisted re-labelling.** The "disturbing" and "restricted" videos can be re-annotated with fine-grained ADD/SXL/PH/HH labels using an LLM applied to their metadata, creating a child-safety-specific version of the fine-grained label space.

## 2. Add an Age-Aware Policy Decision Layer

A future version could extend the binary Safe/Harmful verdict with an age-stratified policy layer. Different age groups have different sensitivity to the same content — what is appropriate for a 12-year-old may not be appropriate for a 4-year-old. An age-aware policy would apply different thresholds per age group and could introduce graduated decisions (e.g. Allow / Warn / Block) tailored to developmental sensitivity rather than a single binary verdict.

## 3. Improve the Explanation Component with XAI

Strengthen the explanation module by using explainable AI techniques that make model outputs easier to interpret and audit. This can help show which inputs most influenced a recommendation and improve trust in the system.

## 4. Evaluate Results Further

Carry out a more comprehensive evaluation of model and policy performance across recommendation categories. This should include quantitative metrics, error analysis, and comparisons between model predictions and final policy decisions on held-out data beyond the MetaHarm validation split.
