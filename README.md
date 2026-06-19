# Computer Vision Final Project Report

# Enhanced Face Swapping Using Fine-Tuning, High-Frequency Enhancement, and Visual Harmonization

**Group 6**

* Jorich Loquero 卓仁智 (314540011)
* Abdul Basith Bahi 張敬輝 (314561004)
* Britney McDonald 唐佩芸 (314540004)
* Naufal Nur Rafi 任光賢 (313540036)
* Febrian Bahari Adi 林海龍 (313540032)
* Admiral Hisyam Zidny 安德穆 (314540001)
* Syahrizal M. Nurjaya 夏里恩 (314540039)

**National Yang Ming Chiao Tung University (NYCU)** *June 2026*

---

## Abstract

This report outlines an enhanced face-swapping pipeline developed to improve upon the standard SimSwap architecture. Face swapping requires a delicate balance between maximizing source identity transfer and preserving target attributes. We present a multi-modular approach integrating balanced metric-guided partial fine-tuning, high-frequency identity-guided adaptive enhancement, visual harmonization, and a dual-layer watermarking system for ethical media traceability. Our experiments demonstrate quantifiable improvements in both identity retrieval metrics and posture error reduction over the baseline.

---

## 1. Introduction

Face swapping is a sophisticated computer vision technique designed to take a target image and replace its facial features with the identity of a source image, all while keeping the target's non-identity attributes (such as expression, pose, lighting, and background) intact. The fundamental challenge in this domain is resolving the conflict between identity transfer strength and attribute preservation.

In our project, we utilize SimSwap [1] as the baseline architecture. SimSwap relies on an ID Injection module to embed the source identity into the generator's feature space and a Feature Matching Loss to preserve the target attributes. However, the baseline study reveals a core limitation: SimSwap deliberately settles for a competitive identity performance (achieving 92.8% ID retrieval compared to competing software reaching 97.4%) in order to prioritize the preservation of target attributes.

**Why it is important to solve:** The authors of SimSwap accepted this lower retrieval percentage because pushing harder on identity transfer degraded attribute preservation. Our goal is to improve the quality of the identity signal itself and optimize the generation process so that the balance point can be moved without sacrificing target attributes. Solving this limitation is vital for generating high-fidelity, highly realistic face swaps that do not suffer from identity degradation or geometric uncanny valley anomalies.

**Method Used:** To answer whether pose-preservation constraints and pose-aware training can improve this balance, we maintain SimSwap’s core swapping architecture and augment it. Our method operates on a multi-stage refinement pipeline:

1. **Identity Enhancement & Training Optimization:** We partially fine-tune the identity-injection parameters using specialized loss constraints.
2. **High-Frequency Enhancement:** We extract and inject high-frequency residuals from super-resolution models to restore structural details.
3. **Visual Harmonization:** We apply targeted corrections to reduce boundary artifacts and harmonize lighting.
4. **Watermark Integration:** We embed invisible cryptographic and visible tracking signals into the final output to ensure media provenance.

---

## 2. Review of Previous Work

Previous state-of-the-art frameworks, including the standard SimSwap architecture, fundamentally depend on generalized facial recognition networks such as ArcFace [2] to extract identity embeddings. While highly effective for general identification, these standard extractors respond to a combination of facial geometry and external appearance characteristics like skin tone, hair, and ambient lighting.

**Limitations of Previous Methods:** This combination leads to a phenomenon known as *identity bypass*. When the generator attempts to minimize the identity loss, the easiest mathematical path is often to copy the source person's appearance (coloration and lighting) into the result rather than transferring the actual structural geometry. Replacing ArcFace with CurricularFace (a stronger variant better at discriminating between similar-looking people) did not resolve the issue; the model still found the appearance-copying shortcut, resulting in a mere lighting adjustment rather than true geometric transfer.

**Why our method is better:** Our enhanced pipeline specifically targets and resolves this bypass through two main avenues. First, through exploration of purely geometric extractors like BlendFace. Second, and more importantly, our finalized approach bypasses the rigid limitations of the pretrained generator by introducing *Balanced Metric-Guided Partial Fine-Tuning*. By freezing the global generator and updating only the AdaIN-related injection parameters under strict edge-preservation and baseline-preservation constraints, we force the network to adapt geometric structure without leaking source appearance.

**Summary of Main Contributions:**

* **Architectural Analysis:** Systematic evaluation of identity encoders (ArcFace, CurricularFace, BlendFace) to highlight the causes of attribute leakage.
* **Partial Fine-Tuning Optimization:** Implementation of a localized update mechanism restricted to modulation parameters to boost ID transfer while maintaining stability.
* **Adaptive High-Frequency Enhancement:** Introduction of a safe, gated high-frequency detail injection module using Real-ESRGAN [3] to recover texture and sharpness.
* **Ethical Traceability Framework:** Integration of a dual-layer steganographic and visual watermarking system to guarantee accountability in synthetic media generation.

---

## 3. Summary of the Technical Solution

Our technical solution decomposes into four distinct structural modifications built upon the SimSwap backbone.

### 3.1 Identity Encoder Exploration

An identity extractor converts a face into an embedding vector. We evaluated three paradigms:

* **ArcFace & CurricularFace:** Both are sensitive to appearance and geometry, leading to identity bypass where skin tone and texture are copied improperly.
* **BlendFace:** Trained on images where appearance is systematically blended out, meaning the embedding only sees face geometry (jaw, cheekbones, nose bridge). Testing showed BlendFace successfully dropped attribute leakage to a negative value, validating that forcing geometry-only translation eliminates identity bypass.

### 3.2 Balanced Metric-Guided Partial Fine-Tuning

SimSwap injects source identity into the target face feature ($f$) using AdaIN-style modulation:

$$\text{AdaIN}(f, \gamma, \beta) = \gamma \cdot \left(\frac{f - \mu(f)}{\sigma(f)}\right) + \beta$$

where $\gamma$ and $\beta$ are the identity modulation parameters derived from the source embedding.

Instead of full discriminator training, we initialized from the official pretrained SimSwap, froze the ArcFace encoder and most generator layers, and fine-tuned *only* the identity-injection parameters. The loss objective is defined as:

$$L_{\text{FT}} = \lambda_{\text{id}}L_{\text{id}} + \lambda_{\text{base}}L_{\text{base}} + \lambda_{\text{edge}}L_{\text{edge}} + \lambda_{\text{self}}L_{\text{self}}$$

* **Identity-ranking loss ($L_{\text{id}}$):** Encourages the output to be closer to the source identity than the target identity.
* **Baseline-preservation loss ($L_{\text{base}}$):** Keeps the fine-tuned output close to the stable pretrained SimSwap output.
* **Edge-preservation loss ($L_{\text{edge}}$):** Preserves facial contours, boundaries, and local structures.
* **Self-reconstruction loss ($L_{\text{self}}$):** Prevents unnecessary changes when the source and target are the identical image.

### 3.3 High-Frequency Identity-Guided Adaptive Enhancement

To improve small facial details (eyes, skin texture) without degrading identity, we utilize a gated High-Frequency (HF) residual pass.

1. Apply Real-ESRGAN to the base swap: $I_{\text{RE}} = \text{RealESRGAN}(I_{\text{base}})$.
2. Compute restoration residual: $R = I_{\text{RE}} - I_{\text{base}}$.
3. Extract HF residual: $R_{\text{HF}} = R - \text{GaussianBlur}(R)$.
4. Generate candidate: $I_{\text{cand}} = \text{clip}(I_{\text{base}} + \alpha R_{\text{HF}}, 0, 1)$.

This candidate is subjected to strict gating:

* **Identity Gate:** Ensures $S_{\text{id}}(I_{\text{cand}}, x_s) \geq S_{\text{id}}(I_{\text{base}}, x_s) - \epsilon$.
* **Drift Gate:** Ensures $D_{L1}(I_{\text{cand}}, I_{\text{base}}) \leq \tau$ to avoid excessive visual change.
* **Sharpness Gate:** Ensures $\text{Sharpness}(I_{\text{cand}}) > \text{Sharpness}(I_{\text{base}})$.

If the candidate fails any gate, the baseline output is preserved.

### 3.4 Visual Harmonization & Dual-Layer Watermarking

Finally, boundaries and lighting discrepancies are resolved through a harmonization pass. To maintain ethical AI generation standards, a dual-layer watermark is applied:

* **Invisible Traceability:** Uses Least Significant Bit (LSB) steganography to embed a hidden tracking signature (technical lineage) directly into the image tensor. It alters pixel color values by a maximum of 1 unit, remaining imperceptible to human sight but mathematically verifiable.
* **Proactive Transparency:** Appends a visible, semi-transparent text indicator ("GROUP_6") to the bottom-right corner.

---

## 4. Experiments

We systematically measured the impact of our modular additions using two primary metrics: Identity (ID) Retrieval percentage and Posture Error.

### 4.1 Quantitative Results

Each stage of our pipeline contributed to the final result, as detailed in the tables below.

**Table 1: ID Retrieval Performance Across Pipeline Stages**

| Pipeline Stage | ID Retrieval (%) $\uparrow$ |
| --- | --- |
| SimSwap Baseline | 80.0 |
| Finetune | 80.3 |
| FT + HF Module | 80.7 |
| **FT + HF + Harmonization (Final)** | **82.0** |

**Table 2: Posture Preservation Across Pipeline Stages**

| Pipeline Stage | Posture Error $\downarrow$ |
| --- | --- |
| SimSwap Baseline | 2.635 |
| Finetune | 2.654 |
| FT + HF Module | 2.706 |
| **FT + HF + Harmonization (Final)** | **2.607** |

While the isolated Fine-Tuning and HF modules slightly increased posture error (as their mathematical optimization prioritizes identity boundaries and texture sharpness), the integration of the Visual Harmonization module pulled the posture error down to $2.607$, achieving a better structural alignment than the original baseline.

### 4.2 Qualitative Results and Watermark Validation

Visual inspections confirm that the target pose and facial expression are heavily preserved while identity characteristics translate far more sharply. Bounding-box and facial landmark tracking show the generated results tightly adhere to the target’s structural limits, reducing boundary artifacts.

Furthermore, we achieved a $100\%$ success rate in watermark extraction logic. During tests with the generated face-swaps, the embedded LSB signatures were extracted successfully, matching the original embedded strings precisely without degrading visual fidelity.

---

## 5. Conclusions

In conclusion, we proposed and successfully implemented a comprehensive refinement pipeline to elevate the quality of face-swapping capabilities. The method leverages three central optimization modules:

1. **Partial Fine-Tuning** improves source identity preservation efficiently by exclusively updating identity-injection parameters via targeted loss functions.
2. **High-Frequency Enhancement** successfully injects critical facial details (such as eye structure, skin texture, and mouth contours) while effectively blocking identity degradation and spatial drift via strict algorithmic gating.
3. **Visual Harmonization** successfully cleans up the final image representation by reducing boundary artifacts and harmonizing skin tones.

Overall, this combined architecture produces a sharper, more visually consistent face-swapping result with stronger ID retrieval and lower pose error compared to the pretrained SimSwap baseline, all while embedding an ethical dual-layer traceability watermark.

---

## 6. References

1. Chen, R., Chen, X., Ni, B., & Ge, Y. (2020). SimSwap: An Efficient Framework for High Fidelity Face Swapping. In *Proceedings of the 28th ACM International Conference on Multimedia*.
2. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
3. Wang, X., Xie, L., Dong, C., & Shan, Y. (2021). Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. In *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*.
4. Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
5. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., & Loy, C. C. (2018). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. In *European Conference on Computer Vision Workshops (ECCVW)*.
6. InsightFace. (n.d.). *InsightFace: 2D and 3D Face Analysis Project*. InsightFace official repository / project page.
