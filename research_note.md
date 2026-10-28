## Research Question

Does the main German public newspaper outlet *Tagesschau* cover East German states through different topics and frames than comparable West German states?

- **H1 (topic agenda).** Articles about East German states are more concentrated in politics, extremism and protest topics, and less in economy, culture and science, than articles about comparable West German states.
- **H2 (deficit framing).** For the same topic, articles about East German states use deficit frames (far-right, economic lag, depopulation) more often.
- **H3 (collective label).** East German states are more often described as a bloc. Articles about them use "ostdeutsch\*" or "der Osten" far more often than articles about West German states use "westdeutsch\*" or "der Westen".
- **H4 (election shock).** The gaps in H1 and H2 widen around the 2024 state elections in Saxony and Thuringia (1 September) and Brandenburg (22 September).

## Data Collection Strategy

- **Data set:** [stefan-it/awesome-tagesschau](https://huggingface.co/datasets/stefan-it/awesome-tagesschau)
- **Time window:** 2023–2025 (coverage before 2022 is low)
- **Wrangling:**
  - Remove unnecessary variables:
    - `images`
    - `teaserImage`
    - `video`
    - `firstFrame`
    - `brandingImage`
    - `breakingNews`
    - `tracking`
    - `regionalId` (only ~60 values assigned)
  - Remove duplicate values
  - Articles are

## NLP Pipeline

1. **Parsing.** Load the JSON, keep stories only, join the text blocks (`title` and `content`) into one article text, `content` is currently an object which has to be joined into one flow text probably. Drop bylines which contain information on the reporting local news broadcaster. Deduplicate data. 
2. **State assignment.** Editor tags naming a state or its election (e.g. "Thüringen", "Landtagswahl Sachsen") are matched exactly first. Untagged articles are matched with a gazetteer of state names and larger municipalities, using spaCy (`de_core_news_lg`) location entities and weighting mentions in the title and teaser higher. The gazetteer step is validated against the tagged articles.
3. **Topic baseline.** Compare the shares of the Tagesschau's own ressort and tags between East and West articles.
4. **Topic model.** BERTopic with German sentence embeddings (`deepset/mxbai-embed-de-large-v1`), fixed seed, about 30–50 topics, labelled by reading representative articles. State and place names are masked first, so topics do not form around regions.
5. **Frame classification.** A six-frame codebook (far-right/extremism, economic deficit, demographic decline, success/transformation, GDR history/reunification, other), applied at passage level. About 300 passages are hand-labelled (100 development, 200 test). Zero-shot NLI with a multilingual DeBERTa model is the main model; a keyword dictionary and TF-IDF with logistic regression are the baselines, compared on macro-F1.
6. **Collective label.** Count mentions of the East and West labels per 1,000 articles about each group of states.
7. **Hypothesis tests.** Regress topic and frame indicators on an East dummy, with month and ressort fixed effects. There are only about 15 states, so inference uses randomisation across states (shuffle the East label among states) rather than clustered standard errors. H4 adds an interaction of East with the July–September 2024 election window.
8. **Figures.**
   1. Topic and frame share differences between East and West with confidence intervals
   2. Deficit-frame share by month, with the 2024 elections marked
   3. The East/West label rates from H3