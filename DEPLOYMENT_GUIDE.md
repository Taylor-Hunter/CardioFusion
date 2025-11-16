# 🚀 CardioFusion Deployment Guide

## 📋 Overview

This guide provides step-by-step instructions for deploying the CardioFusion web application to Streamlit Cloud.

---

## 🔧 Prerequisites

Before deploying, ensure you have:

- ✅ GitHub account
- ✅ Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- ✅ All models trained and saved
- ✅ Repository pushed to GitHub

---

## 📦 Pre-Deployment Checklist

### 1. Verify File Structure

```
ml-cardio/
├── app.py                    ✅ Main application
├── requirements.txt          ✅ Dependencies
├── .streamlit/
│   └── config.toml          ✅ Configuration
├── utils/                    ✅ Utility modules
│   ├── __init__.py
│   ├── model_utils.py
│   ├── shap_explainer.py
│   └── data_validator.py
├── models/                   ✅ Trained models
│   ├── baseline_models/
│   └── advanced_models/
└── README.md                 ✅ Documentation
```

### 2. Update .gitignore

Ensure your `.gitignore` includes:

```
# Data files
*.csv
!requirements.txt

# Model files (if too large for GitHub)
# models/
# *.pkl
# *.h5

# Environment
.env
.venv/
venv/
```

**Note**: If your model files are too large (>100MB), you'll need to use Git LFS or regenerate them after deployment.

---

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Complete CardioFusion project with Streamlit app"
git push origin claude/project-review-012ZuoZfFzZJs84MgLDQwTMb
```

### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `Apc0015/ml-cardio`
4. Select branch: `claude/project-review-012ZuoZfFzZJs84MgLDQwTMb`
5. Main file path: `app.py`
6. Click "Deploy"

### Step 3: Configure Advanced Settings (Optional)

In Streamlit Cloud advanced settings:

- **Python version**: 3.10
- **Secrets**: Add any required secrets (none needed for basic deployment)

---

## 🔄 Handling Large Model Files

If model files exceed GitHub's 100MB limit:

### Option 1: Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pkl"
git lfs track "*.h5"

# Add and commit
git add .gitattributes
git add models/
git commit -m "Add models with Git LFS"
git push
```

### Option 2: Regenerate Models on Deployment

Create a `setup.sh` script that runs after deployment:

```bash
#!/bin/bash

# Run preprocessing
jupyter nbconvert --to notebook --execute data_preprocessing.ipynb

# Run baseline models
jupyter nbconvert --to notebook --execute baseline_models.ipynb

# Run advanced models
jupyter nbconvert --to notebook --execute advanced_models.ipynb
```

Add to `.streamlit/config.toml`:

```toml
[server]
headless = true
```

---

## 🧪 Local Testing Before Deployment

Test your app locally first:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

The app should open at `http://localhost:8501`

---

## 🐛 Troubleshooting

### Issue: Module Not Found

**Solution**: Ensure all dependencies are in `requirements.txt`

```bash
pip freeze > requirements.txt
```

### Issue: Models Not Loading

**Solution**: Check model file paths in `app.py`

```python
# Ensure paths are relative, not absolute
predictor = ModelPredictor('models')  # Not '/home/user/models'
```

### Issue: Memory Limit Exceeded

**Solution**: Reduce model size or use smaller background data for SHAP

```python
# In app.py, reduce SHAP background sample
background_data = X_train.sample(min(50, len(X_train)), random_state=42)
```

---

## 📊 Post-Deployment

After successful deployment:

1. ✅ Test all functionality
2. ✅ Verify predictions work correctly
3. ✅ Check SHAP explanations
4. ✅ Test both simple and detailed views
5. ✅ Share the URL with stakeholders

---

## 🔐 Security Best Practices

- ❌ Never commit API keys or secrets
- ✅ Use Streamlit secrets for sensitive data
- ✅ Sanitize all user inputs
- ✅ Include medical disclaimers

---

## 📞 Support

For deployment issues:
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)

---

**Happy Deploying!** 🚀
