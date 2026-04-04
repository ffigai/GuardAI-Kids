# Future Work

## 1. Replace MetaHarm with a Child-Specific Training Dataset

The most impactful improvement for a future version is replacing MetaHarm with a dataset designed specifically for child safety on video platforms. MetaHarm is a general-purpose harmful content dataset built for broad adult audiences — it was not designed around what is inappropriate *for children*, and this causes two structural problems:

- **Missing child-appropriate negatives.** MetaHarm has no verified examples of child-safe content. The model has never learned that nursery rhyme vocabulary, animal nature content, or cartoon thumbnails are safe signals, so it misclassifies mainstream children's videos (see Known Limitations).
- **Label contamination from bad actors.** Harmful content that mimics children's vocabulary (adult parodies, inappropriate remixes using "nursery rhyme" or "kids songs" as cover) is labelled harmful in MetaHarm. The model learns these words as risk signals, penalising legitimate children's content.

The recommended dataset for a future version is the **Disturbed YouTube for Kids** dataset (Papadamou et al., ICWSM 2020), available at [https://zenodo.org/records/3632781](https://zenodo.org/records/3632781). It contains 4,797 manually annotated YouTube videos labelled as suitable, disturbing, restricted, or irrelevant — built specifically for detecting inappropriate content in the children's video space.

Practical angles for using this dataset:

- **Add "suitable" videos as hard negatives.** The "suitable" class provides verified child-appropriate content that is currently missing from training. Adding these alongside MetaHarm harmful examples would directly fix the false positive patterns observed on Cocomelon and nature videos.
- **Use as a held-out evaluation benchmark.** Evaluating the current model on the Disturbed YouTube for Kids dataset (mapping suitable → ALLOW, disturbing/restricted → WARN/BLOCK) would quantify how poorly the model generalises to real-world children's YouTube content — a meaningful finding for the paper.
- **LLM-assisted re-labelling.** The "disturbing" and "restricted" videos can be re-annotated with fine-grained ADD/SXL/PH/HH labels using an LLM (e.g. Claude, GPT-4) applied to their metadata, creating a child-safety-specific version of the fine-grained label space.

## 2. Improve the Age-Aware Policy Decision Maker

Refine the policy layer so it better captures differences between age groups and handles ambiguous cases more consistently. Future versions could use more structured thresholds, risk weighting, and policy rules tailored to developmental sensitivity.

## 3. Improve the Explanation Component with XAI

Strengthen the explanation module by using explainable AI techniques that make model outputs easier to interpret and audit. This can help show which inputs most influenced a recommendation and improve trust in the system.

## 4. Evaluate Results Further

Carry out a more comprehensive evaluation of model and policy performance across age groups and recommendation categories. This should include quantitative metrics, error analysis, and comparisons between model predictions and final policy decisions on the MetaHarm-based XLSX training data.

---

## Known Limitations

### False positives on animal/nature content

**Example:** "Turkey vs birds" (https://www.youtube.com/watch?v=zoC-GJLCRCs)

This is a harmless nature video showing a turkey interacting with birds. All three models flag it incorrectly:

- **Text model** — PH score 0.776 (Warn). The word "vs" in a combative framing inflates the Physical Harm signal. The model has no understanding that animal behaviour is a distinct context from human violence.
- **Image model** — ADD score 0.663 (Block). The thumbnail's visual style (bright colours, action composition) resembles thumbnails associated with addictive/clickbait content in the training data. This is a visual false positive driven by thumbnail aesthetics rather than genuine harmful content.
- **Multimodal model** — PH score 0.922 (Block). The fusion of "vs" in the title and the action-style thumbnail amplifies the Physical Harm signal dramatically, producing a high-confidence incorrect Block recommendation.

**Root cause:** The training data likely lacks sufficient examples of animal/nature content with combative framing. The models conflate animal conflict with human physical harm, and attention-grabbing thumbnail aesthetics with addictive content. This category of video — nature, wildlife, animal behaviour — represents a systematic blind spot.

**Implication for use:** The system should not be used as a sole decision-maker for nature and wildlife content without human review. A post-processing filter or a dedicated "nature/animal" context classifier could mitigate this class of false positive in future work.

### Spurious SXL association with "nursery" / children's song vocabulary

**Example:** "Wheels on the Bus — CoComelon Nursery Rhymes & Kids Songs" (https://www.youtube.com/watch?v=e_04ZrNroTo)

A well-known Cocomelon nursery rhyme video receives a Warn recommendation for SXL across all age groups:

- **Text model** — SXL=0.670. Gradient attribution shows the top tokens driving the SXL score are "nursery", "Nurs", "Kids", "Rhymes", "Lyrics". The model has learned a spurious association between these words and sexual content, likely because the training data contains harmful videos (adult parodies, inappropriate remixes) that use children's song titles and the word "nursery" as cover. The model cannot distinguish this innocent context from those harmful ones.
- **Image model** — ADD=0.894 (Block). Cocomelon's hyper-colourful, animated thumbnail triggers the addictive/engagement-bait classifier. While not entirely incorrect (Cocomelon has been criticised by child psychologists for overly stimulating visuals), a Block recommendation is disproportionate for a mainstream children's channel.
- **Multimodal model** — SXL=0.543. The text SXL signal carries through, slightly reduced because the thumbnail provides no visual SXL cues.

**Root cause:** The word "nursery" co-occurs with SXL-labelled content in the training set often enough that the model treats it as a risk signal. This is a training data contamination issue — bad actors using children's content vocabulary as camouflage creates a false association that penalises legitimate children's content.

**Implication for use:** Mainstream children's channels (Cocomelon, Peppa Pig, etc.) and any video with "nursery rhyme" in its metadata are at systematic risk of false SXL flags. A trusted-channel whitelist or a dedicated children's content context classifier would be needed to address this in production.
