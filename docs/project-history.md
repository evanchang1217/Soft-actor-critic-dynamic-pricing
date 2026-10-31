# Project history and implementation scope

The original capstone poster credits **張凱盛** as the project member and **陳朝鈞** as the advisor at National Cheng Kung University. The repository records three different layers of work; their results and implementation scope should be kept distinct.

| Record | Content and evidence |
| --- | --- |
| Original capstone | Poster describing SAC-based game-economy control, historical-data behavior modeling, comparative simulation figures, a web demonstration, and a daily server price-update workflow |
| Released demo before maintenance | Public `myapp/sac_model/sac_component.py`, Django configuration/training/status pages, two custom Minecraft status API reads, and stored demo figures |
| Current maintenance and follow-up | Shared research modules, correctness fixes and explicit modeling changes, reproducible experiment tools, new baselines and fee ablation, reward analysis, saved results/checkpoints, tests, and updated local demonstration |

The pre-maintenance implementation is available at [commit e092c86](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/tree/e092c8684b2e42dd0277d053d38824a194e51aac). Original poster figures are preserved under [`results/original/`](../results/original/); all newly executed comparisons are separately identified under [`results/reevaluation/`](../results/reevaluation/).

The current simulator derives from the released demo's heuristic price/inventory/budget behavior model. It does not recover the original historical-data fitting process. The new baselines, model settings, validation/test protocol, and scenario checks should be described as new evaluation work, not as the exact original experiments.

The currently released Minecraft integration reads market and player status. The original scheduled inference/write-back code, server plugin, historical dataset, and deployed checkpoint are not included. The poster documents the project-level system; the public adapter by itself does not demonstrate the complete deployed workflow.

The poster establishes the original project attribution. A detailed per-component account of individual implementation responsibilities needs confirmation from the project author; it cannot be inferred just from a file being present in this repository. When describing this project in an application, distinguish original design/implementation responsibilities from later maintenance and re-evaluation.
