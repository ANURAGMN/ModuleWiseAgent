# Git Commit Guide for Local Machine

## Your Local Setup
- Location: `C:\Users\anurag.mn\ModuleWiseAgent`
- Files to commit: `tfidf_rag_agent.py` and any example scripts

---

## Step-by-Step Commit Process

### Step 1: Check Status

```powershell
# In PowerShell at C:\Users\anurag.mn\ModuleWiseAgent
git status
```

This will show you all new/modified files.

---

### Step 2: Review Changes

```powershell
# See what's changed
git diff

# List untracked files
git ls-files --others --exclude-standard
```

---

### Step 3: Add Files

```powershell
# Add specific file
git add tfidf_rag_agent.py

# Or add all Python scripts
git add *.py

# Or add everything
git add .
```

---

### Step 4: Create Commit

```powershell
git commit -m "Add TF-IDF RAG agent for lightweight PDF analysis

- Implement TF-IDF based document embeddings (no heavy dependencies)
- Use pdfplumber for PDF extraction (works without admin access)
- Add scikit-learn for vector similarity search
- Support interactive CLI mode with debug capabilities
- Include example scripts for automated analysis
- No API keys required, fully local operation
- Resolves DLL dependency issues on Windows without admin access"
```

---

### Step 5: Push to Remote (Optional)

```powershell
# Push to your branch
git push origin cursor/document-analysis-agent-enhancement-7c74

# Or just
git push
```

---

## Quick One-Liner

```powershell
git add tfidf_rag_agent.py && git commit -m "Add lightweight TF-IDF RAG agent for PDF analysis" && git push
```

---

## Files You Should Commit

Based on your work:
- ✅ `tfidf_rag_agent.py` (main agent)
- ✅ Any example scripts you created:
  - `analyze_reports.py`
  - `generate_report.py`
  - `export_analysis.py`
  - `daily_insights.py`
  - etc.

---

## Check What Will Be Committed

```powershell
# Dry run - see what would be committed
git add --dry-run .

# See staged changes
git diff --cached
```

---

## Commit Message Templates

### For main agent file:
```
Add TF-IDF based RAG agent for PDF analysis without dependencies

- Uses pdfplumber instead of PyMuPDF (no C++ dependencies)
- TF-IDF embeddings via scikit-learn (no torch/transformers)
- Works on Windows without admin access
- Interactive CLI with debug mode
- Fully local, no API keys needed
```

### For example scripts:
```
Add automated analysis scripts for RAG agent

- Report generation (Markdown/JSON/CSV)
- Batch processing examples
- Daily insights automation
- Comparative analysis tools
```

---

## Troubleshooting

### Issue: "Please tell me who you are"

```powershell
git config --global user.email "anurag.mn@redbus.com"
git config --global user.name "Anurag M N"
```

### Issue: "Permission denied"

Check if you're on the right branch:
```powershell
git branch
# Should show: * cursor/document-analysis-agent-enhancement-7c74
```

### Issue: Large files warning

Add to `.gitignore`:
```
*.pdf
venv/
__pycache__/
*.pyc
```

---

## Best Practices

1. **Commit frequently** - Don't wait for perfect code
2. **Write clear messages** - Explain what and why
3. **Review before commit** - Use `git diff` to check
4. **Test before push** - Make sure code runs
5. **Keep commits focused** - One feature per commit

---

## After Committing

```powershell
# View your commit
git log -1

# View commit with changes
git show HEAD

# Check remote status
git status
```
