# Feature Activation Visualization (s3_feature_activation_graphs)

This directory contains the third stage of the SAE feature discovery pipeline: visualizing temporal activation patterns of top-ranked features as tokens are progressively added to sentences. This stage takes ranked features from **s2_find_top_features** and creates token-by-token visualizations that reveal when and how features activate during sentence processing.

## Overview

This stage visualizes feature activations by:
- **Loading top-N ranked features** from interpretability analysis
- **Extracting token-level activations** from sentence-wise NPZ files
- **Plotting activation trajectories** showing feature strength across token positions
- **Generating summary statistics** on temporal activation patterns (early vs. late activators)
- **Organizing visualizations** by concept label for easy comparative analysis

The output enables qualitative assessment of feature interpretability through visual inspection and pattern recognition across label categories.

## Motivation

While quantitative metrics (correlation, monotonicity, selectivity) provide objective feature rankings, **temporal activation patterns** reveal crucial interpretability insights that metrics alone cannot capture:

1. **Keyword vs. Context Detection**: Do features activate immediately upon seeing specific keywords ("lied"), or do they require surrounding context?

2. **Incremental Understanding**: How does feature confidence build as more tokens are revealed? Gradual vs. sudden activation patterns suggest different computational mechanisms.

3. **Semantic Completeness**: Do features require full sentence semantics, or can they reliably detect the target concept from partial information?

4. **Label Discrimination**: Visual comparison across labels (0-3) reveals whether activation strength genuinely scales with concept intensity or shows spurious patterns.

5. **Validation of Metrics**: Visualizations provide sanity checks on quantitative rankings—features with high interpretability scores should show clear, interpretable temporal patterns.

## Methodology

### Activation Trajectory Analysis

Rather than analyzing only final activations (complete sentences), we visualize **cumulative activation trajectories** across token positions:

```
Sentence: "I lied to my boss"
Tokens:   ["I", "lied", "to", "my", "boss"]

Feature activations visualized at each step:
  Token 1 ("I"):              [activation values for top-N features]
  Token 2 ("I lied"):         [activation values]
  Token 3 ("I lied to"):      [activation values]
  Token 4 ("I lied to my"):   [activation values]
  Token 5 ("I lied to my boss"): [activation values]
```

This reveals:
- **Early activators**: Spike at first mention of relevant keywords
- **Contextual activators**: Gradual increase requiring multiple context words
- **Late activators**: Minimal activation until final tokens, needing complete semantics

### Visualization Design Principles

**One plot per sentence approach:**
- Enables detailed examination of individual examples
- Facilitates comparison across different sentences with same label
- Allows inspection of edge cases and failure modes
- Supports manual validation of algorithmic feature rankings

**Token-based X-axis:**
- Displays actual token strings rather than positional indices
- Provides immediate readability—no need to reference original sentence
- Enables direct connection between lexical content and activation patterns
- Supports linguistic analysis of activation triggers

**Multi-feature overlay:**
- Visualizes top-N features simultaneously on one plot
- Reveals relative activation strengths across different features
- Shows whether features activate synchronously or sequentially
- Enables comparison of complementary vs. redundant feature patterns

## Visualization Interpretation Framework

### Activation Pattern Categories

**Early Activators** (keyword detectors)
- **Signature**: Rapid spike at first few tokens
- **Interpretation**: Feature detects explicit lexical markers
- **Example**: Dishonesty feature activating immediately on "lied"
- **Strength**: High precision when keyword present
- **Weakness**: May miss implicit or paraphrased instances

**Contextual Activators** (compositional understanding)
- **Signature**: Gradual increase across multiple tokens
- **Interpretation**: Feature integrates multiple context words
- **Example**: Sarcasm detector requiring both setup and punchline
- **Strength**: Captures relationships between words
- **Weakness**: Requires more tokens before confident prediction

**Late Activators** (semantic comprehension)
- **Signature**: Flat until final tokens, then sudden increase
- **Interpretation**: Feature requires complete sentence semantics
- **Example**: Implied dishonesty needing full context to disambiguate
- **Strength**: Handles complex, context-dependent cases
- **Weakness**: Cannot make early predictions

### Label-Wise Comparison

Visual inspection across labels reveals interpretability quality:

**Strong interpretability indicators:**
- Label 3 (strong): Multiple features show high activation
- Label 2 (moderate): Features activate but at lower magnitude
- Label 1 (weak): Minimal or inconsistent feature activation
- Label 0 (neutral): All features remain near-zero

**Problematic patterns suggesting poor interpretability:**
- High activation on label 0 sentences (false positives)
- No activation on label 3 sentences (false negatives)
- Erratic activation unrelated to label strength
- Identical patterns across all labels (feature doesn't discriminate)

## Key Design Decisions

1. **Sentence-level granularity**: One plot per sentence rather than aggregated views enables detailed examination of individual cases and edge case identification.

2. **Token-based axis**: Actual token strings provide immediate interpretability without requiring separate sentence reference, supporting linguistic analysis.

3. **Final-token extraction**: While visualizations show all tokens, statistical analysis focuses on final tokens to avoid data leakage from cumulative fragments.

4. **Label-based organization**: Grouping plots by concept label (0-3) facilitates systematic comparison across label categories.

5. **Summary statistics**: Automated pattern detection (early vs. late activators, peak positions) supplements manual visual inspection.

## Integration with Pipeline

**Inputs required:**
1. **From s2**: `ranked_features.csv` with top interpretable features (layer, feature_id, scores)
2. **From s1**: `sentence_*.npz` files with token-level activation trajectories

**Pipeline position:**
```
s1_compute_all_features → s2_find_top_features → s3_feature_activation_graphs → s4_classifier_data_prep
         ↓                         ↓                         ↓                          ↓
  Activation data           Feature ranking          Visual validation         Training data
```

**Typical workflow:**
1. Extract activations (s1) - expensive, one-time GPU computation
2. Rank features (s2) - fast, iterative parameter tuning
3. Visualize top features (s3) - qualitative validation via plots
4. Prepare classifier data (s4) - informed by visualization insights

## Usage

For detailed technical instructions and CLI examples, see [USAGE.md](USAGE.md).

**Quick start:**
```bash
cd s3_feature_activation_graphs

uv run python visualize_activations.py \
  --s2-results ../s2_find_top_features/results/dishonesty_100_results \
  --s1-data ../s1_compute_all_features/dishonesty_100_features \
  --output ./plots \
  --top-n 10
```

## Output Structure

```
s3_feature_activation_graphs/
├── visualize_activations.py
├── plotting_utils.py
├── README.md (this file)
├── USAGE.md
└── plots/
    ├── label_0/        # Neutral sentences (label=0)
    │   ├── sentence_000001.png
    │   ├── sentence_000005.png
    │   └── ...
    ├── label_1/        # Weak instances (label=1)
    ├── label_2/        # Moderate instances (label=2)
    ├── label_3/        # Strong instances (label=3)
    └── summary_statistics.txt
```

## Limitations

1. **Manual inspection required**: Visualizations provide qualitative insights but require human interpretation—not fully automated.

2. **Scalability**: Generating plots for 100 sentences × 10 features produces 100 PNG files. Analyzing thousands of sentences becomes unwieldy.

3. **Token-level granularity**: Visualizations show full token sequences, but actual interpretability may depend on subword tokenization artifacts.

4. **Static plots**: Matplotlib PNGs lack interactivity. Investigating specific activation points requires regenerating plots or examining raw data.

5. **Cumulative fragments**: Token-by-token view uses cumulative text fragments (not isolated tokens), which may conflate positional and contextual effects.

## Next Steps

After generating visualizations:

1. **Manual validation**: Browse `plots/label_X/` folders to identify clear activation patterns
2. **Neuronpedia cross-reference**: Use summary statistics URLs to validate features on Neuronpedia
3. **Feature selection**: Note which features show interpretable temporal patterns for downstream use
4. **Refinement**: Adjust s2 thresholds if visualizations reveal poor feature quality
5. **Proceed to s4**: Use validated features and activation patterns to design classifier inputs

## References

- **s1_compute_all_features**: Source of token-level activation data
- **s2_find_top_features**: Source of ranked feature candidates
- **Neuronpedia**: Interactive SAE feature explorer for validation
- **Gemma Scope**: SAE architecture providing interpretable features
