# GitHub Upload

Create an empty GitHub repository without a generated README, license, or
`.gitignore`. Then run:

```bash
cd "/path/to/DE-VJEPA_Independent_Verification"
chmod +x publish_to_github.sh
./publish_to_github.sh https://github.com/USERNAME/REPOSITORY.git
```

Equivalent manual commands:

```bash
python verify_release.py
git init
git add .
git diff --cached --check
git commit -m "Release DE-VJEPA independent verification package"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Do not commit `data/`, `outputs/`, `checkpoints/`, raw per-sample result files,
virtual environments, LaTeX build files, or credentials. Publish large research
artifacts separately with stable URLs and SHA-256 checksums.
