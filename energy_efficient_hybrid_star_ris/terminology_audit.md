# Paper 中同义词 / 近义术语清单

本表列出 manuscript 中可能被混用的技术术语，并给出建议统一用法。

| 概念 | 建议统一用法 | 备注 |
|---|---|---|
| Hybrid STAR-RIS architecture | hybrid active-passive STAR-RIS | Use STAR-RIS when the two-side surface is meant. Use RIS only for general background or cited prior work. |
| Active element set | active set, side-specific active set | “Side support” can be translated as “侧向激活支撑集”, but “某一侧的激活元素集合” is clearer. |
| Sparse active selection | sparse active selection, active-set selection | Use “sparse active selection” in the title/abstract and “active-set selection” for the combinatorial operation. |
| Rate allocation | rate splitting, side-rate splitting | The title/section wording now uses “rate splitting”. |
| EE-SE boundary | achievable EE-SE boundary, Pareto-frontier approximation | Keyword remains “Pareto optimization”. |
| Energy-efficiency metric | EE | The subscript “norm” has been removed. |
| SE-oriented gain | SE-oriented gain law | “Common” was removed from the contribution phrase. |
| EE-oriented gain | EE-oriented gain refinement, power-minimizing gain | “Refinement” means a power-aware adjustment of the SE-oriented gain. |
| Gain variable | side-wise gain \(\beta_s\) | “Common” is used only where all active elements on the same side share \(\beta_s\). |
| Passive gain | \(\beta_s=1\), passive-gain point | Use \(\beta_s=1\) when the mathematical condition matters. |
| Active noise | amplifier noise, active-noise term | “Amplifier noise” is the source. “Active-noise term” is the received/model term. |
| Hardware power | hardware power, switching power, gain-dependent amplification power | Keep the four power components distinct. |
| Total power | total consumed power, \(P_{\mathrm{tot}}\) | Use \(P_{\mathrm{tot}}\) for formulas. |
| BS power | BS transmit power, PA-normalized BS transmit-power term | Distinguish radiated transmit power from PA-normalized consumption. |
| Binary decision | binary active-selection structure | The exact 0--1 program is not solved. |
| Candidate design | candidate active-set pair, boundary point | Use “boundary point” after feasibility and minimum-power selection. |
| Ranking score | cost-aware score \(\xi_{n,s}\) | “Ranker” is informal and is minimized in the revised text. |
| Greedy baseline | channel-strength score, strongest-greedy baseline | The score is \(\pi_{n,s}\). |
| Side notation | transmission side and reflection side | Use \(s\in\{t,r\}\) in formulas. |
| ES protocol | energy-splitting (ES) protocol, ES split | ES protocol is the protocol name. Power splitting refers to \(\rho_s\). |
| Side target | side target \(R_s\), per-user target \(R_s/K_s\) | The equal split is a QoS allocation convention. |
| Element channel strength | BS-to-element gain, scalar BS-to-element gain, stream-averaged effective magnitude | stream-averaged BS-to-element gain \(|g_n|\) | Use \(|g_n|\) after definition. |
| Practical implications | practical design rules | “Transferable” was replaced because it overstates portability. |
