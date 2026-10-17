# Related work (week-0 sweep, 2026-10-17)

Verdict: no published work applies multiple-systems estimation over witnesses
to count unlogged agent actions, with or without adversarial bounds. Each
piece exists on its own. About 80% confidence; paywalled ACM/IEEE coverage
was limited to web search.

## Motivation
- Qin et al., *LLM Agents Can Easily Tamper With Their Own Traces*, 2026, [arXiv 2609.30266](https://arxiv.org/abs/2609.30266). Agents delete, insert and spoof trace events when asked, and unprompted when it raises reward. Proposes an append-only interception server, unevaluated. Says interception does not prove tool calls ran as reported. Never estimates what is missing.
- Arslan, *Audit Without Verification*, 2026, [arXiv 2609.07680](https://arxiv.org/abs/2609.07680). LLM accountability layers relay agents' conclusions. The auditor needs evidence independent of the claims it checks. This motivates H3 and the W4 witness.
- Transluce, [o3 truthfulness](https://transluce.org/investigating-o3-truthfulness), 2025. Models claim tool actions they never took, so the self-report witness has false positives. Linkage has to handle fabricated records.

## Closest work
- Kansal, [label-error-capture-recapture](https://github.com/ParitKansal/label-error-capture-recapture), 2026, repo only. Capture-recapture over LLM judges for label errors. Intervals were valid but too wide to beat a raw count. This is the nearest AI precedent, and a negative one.
- Agent Flight Recorder, 2026, [arXiv 2609.01931](https://arxiv.org/pdf/2609.01931). Hash chains and Merkle commitments detect edits in one log, but not a forked chain.
- VET Your Agent, 2025, [arXiv 2512.15892](https://arxiv.org/abs/2512.15892). Integrity through TEEs and notarised TLS. Not a completeness estimate.
- Hua et al., *Combining Cost-Constrained Runtime Monitors*, NeurIPS 2025, [arXiv 2507.15886](https://arxiv.org/abs/2507.15886). Combines monitors for detection. No population estimate.
- Redwood, [collusion with untrusted monitors](https://redwoodresearch.substack.com/p/how-to-prevent-collusion-when-using). Motivates the L2 adversary.
- PeerReview, SOSP 2007. Witnesses replay tamper-evident logs. Deterministic, no statistics.
- Paccagnella et al., KennyLoggings (CCS 2020) and Custos (NDSS 2020). Attackers suppress audit events through races; the defences add tamper evidence only.

## Capture-recapture outside agents
- Software inspection: Eick et al., ICSE 1992; Vander Wiel & Votta, TSE 1993; Petersson et al., [JSS 2004](https://www.sciencedirect.com/science/article/abs/pii/S0164121203000906); Chun, [EJOR 2006](https://www.sciencedirect.com/science/article/abs/pii/S0377221705005254). In all of these, inspector independence is the known failure point.
- Fuzzing: Böhme, STADS, [arXiv 1803.02130](https://arxiv.org/abs/1803.02130). See also [arXiv 2304.09259](https://arxiv.org/pdf/2304.09259), which argues that deduplication and dependence break capture-recapture in fuzzing.
- Phishing feeds: Moore & Clayton, [2007](https://www.sei.cmu.edu/documents/274/2007_019_001_51383.pdf).

## Methods
- MSE overview: [Bird & King 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6055983/); [HRDAG notes](https://hrdag.org/tech-notes/basic-mse.html).
- Latent class: [Manrique-Vallier 2016 (LCMCR)](https://dmanriqu.pages.iu.edu/preprints/lcm_CR.pdf); identifiability, [Aleshin-Guendel 2020](https://arxiv.org/abs/2008.09865).
- Identification and bounds: [Aleshin-Guendel, Sadinle & Wakefield 2024](https://arxiv.org/abs/2101.09304) (N is not identified without an untestable assumption, so separate the observed-data model from that assumption); [Sun et al.](https://arxiv.org/abs/2008.00127) (confidence sets for N from bounded pairwise dependence); [Das, Kennedy & Jewell](https://arxiv.org/abs/2104.14091); [Dulce Rubio & Kennedy](https://arxiv.org/abs/2407.03539) (lists "not arbitrarily dependent" within the part the others missed).

## Consequences for the design
- If W3 is truly independent of the joint capture by W1 and W2, then Lincoln-Petersen of W3 against the union of the others identifies N, and the result is a point estimate, not bounds. Bounds come back when W3's detection rate varies by action type in a way correlated with the others (e.g. purely local actions that neither the env diff nor the network sink sees). The pre-registration states the assumption in exactly these terms.
- Positive dependence can only support a lower bound (Chao). The direction of the bias is known even when its size is not.
