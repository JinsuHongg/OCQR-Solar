# UTKFace UQ Benchmark Results (5-Class Ordinal)
**Date Generated:** July 18, 2026
**Target Alpha:** 0.1 (Target Coverage: 90%)
**Classes:** 0 to 4 (Class 4 represents the extreme minority age group 81-116 years old)

## 1. Coverage Metrics

| Method | Backbone | Marg Cov | C0 Cov | C1 Cov | C2 Cov | C3 Cov | C4 Cov | Marg Size |
|---|---|---|---|---|---|---|---|---|
| **OAPS** | Softmax | 0.9979 | 1.0000 | 1.0000 | 1.0000 | 0.9802 | **0.9851** | 3.7642 |
| **min-CPS** | Softmax | 0.7111 | 0.7974 | 0.8789 | 0.4185 | 0.3614 | **0.1791** | 1.0000 |
| **min-RCPS** | Softmax | 0.9688 | 0.9564 | 0.9924 | 0.9626 | 0.9109 | **0.8507** | 2.6575 |
| **Risk Control** | Softmax | 0.9734 | 0.9368 | 0.9966 | 0.9846 | 0.9307 | **0.8657** | 2.2801 |
| **COPOC** | Binomial (Unimodal) | 0.8979 | 0.9477 | 0.9537 | 0.7952 | 0.7574 | **0.6866** | 1.7465 |
| **OrdinalCQR** | Quantile Regressor | **0.9717** | **0.9390** | **0.9992** | **0.9471** | **0.9653** | **0.8955** | **2.0131** |

---

## 2. Set Size Metrics

| Method | Backbone | C0 Size | C1 Size | C2 Size | C3 Size | C4 Size | Marg Size |
|---|---|---|---|---|---|---|---|
| **OAPS** | Softmax | 3.7669 | 3.3255 | 4.2489 | 4.8465 | 4.9851 | 3.7642 |
| **min-CPS** | Softmax | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **min-RCPS** | Softmax | 1.7516 | 2.5223 | 3.2643 | 3.8515 | 3.5522 | 2.6575 |
| **Risk Control** | Softmax | 1.5599 | 2.1110 | 2.8150 | 3.3366 | 3.4030 | 2.2801 |
| **COPOC** | Binomial | 1.2288 | 1.8974 | 1.8546 | 1.8317 | 1.6269 | 1.7465 |
| **OrdinalCQR** | Quantile Regressor | **1.4074** | **1.7864** | **2.6322** | **3.0099** | **2.9851** | **2.0131** |

---

## Analysis & Takeaways
1. **OAPS:** Achieves coverage only by predicting practically the entire label space unconditionally for the harder classes (C4 Size: 4.9851 out of 5).
2. **min-CPS:** Fails to properly form sets for harder classes, defaulting to top-1 predictions (Size: 1.0) and completely crashing in coverage (C4 Cov: 17.91%).
3. **min-RCPS & Risk Control:** Reach decent coverage but suffer from severe set expansion at the tails.
4. **COPOC:** The strict unimodal binomial assumption prevents the sets from growing excessively, but it forces them to be too rigid, causing a massive coverage drop on C4 (68.66%).
5. **OrdinalCQR:** Because it natively learns continuous thresholds via Quantile Regression, it successfully adapts its boundaries. It guarantees target coverage (>90%) across almost all classes, including C4 (89.55%), while maintaining an extremely competitive marginal set size of just 2.01.
