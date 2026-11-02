# Current DE-VJEPA manuscript

**DE-VJEPA: Decision-Equivalent Latent Prediction for Policy Fidelity and Control**

- [Paper PDF](DE-VJEPA.pdf)
- [Editable LaTeX source and anonymous supplement](DE-VJEPA_source.zip)
- [Title, abstract, file hashes, and provenance metadata](manuscript_metadata.json)
- [Checksums for the 87 released implementation files](released_implementation_checksums.csv)

The paper uses the TMLR review format, with **12 pages of main content** and
references beginning on page 13. References and appendices bring the complete
PDF to 36 pages. DE-VJEPA remains in the title, and the figures use the
manuscript's Latin Modern Roman font.

## Scope of this revision

The method equation, Figure 2, and Algorithm 1 describe the released
deterministic `h=0` prediction path and separate posterior KL penalty.
The original result labels remain for traceability. The paper does not
claim that this implementation optimizes the full Variational JEPA bound.

The numerical tables and all 53 original compact result CSVs are unchanged.
The new PushT worked example is a separate illustrative rerun, explicitly
identified in the caption and saved figure data. It is not a replacement for
the original aggregate studies. No original aggregate experiment or bootstrap
interval was regenerated for this manuscript revision.

Essential background, all seven core figures, findings, and main limitations
remain in the main text. Appendix H contains the full variational formulation
and extended related work. Appendix B includes the complete artifact statement
and provenance limits. Original raw run logs and checkpoints were not recovered,
so the exact source executed for each historical run cannot be established.

## Build the manuscript

From the repository root, extract the self-contained source bundle into a
separate working directory:

```bash
python -m zipfile -e paper/DE-VJEPA_source.zip manuscript_build
cd manuscript_build
tectonic -k --keep-logs main.tex
```

All figure inputs and compiled vector figures are included. Building the PDF
does not run training. The extracted `README.txt` explains how to edit the
sections and rebuild figures, including their Cairo/font dependencies.
The extracted `implementation/` is an unchanged snapshot of the released
sources plus a documented loss-path audit. The live experimental sources in
this repository remain under `src/devjepa/`.

## Verify the update

```bash
python tools/audit_training_path.py
python tools/verify_paper.py
```

Run these commands from the repository root with the dependencies listed in
the main README. The audits check source integrity and document consistency;
they do not establish independent reproduction of the historical experiments.
