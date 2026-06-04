# DecodeLabs-Task3-OdinakaRitaNwankwo
Project 3:  AI Recommendation Logic

---

## 📌 Overview
This project builds an **AI-powered Tech Stack Recommender** that maps a user's skills to the most relevant career paths using **TF-IDF vectorization** and **Cosine Similarity**. It is a practical implementation of Content-Based Filtering — the same technology that powers Netflix, Spotify, and Amazon recommendations.

---

## 🎯 Goal
Create a recommendation system that takes a user's skills as input and returns the top 3 most suitable career paths, complete with match scores, skill gap analysis, and a visual ranking chart.

---

## ✅ Key Requirements Met
- ✔️ Takes user input (minimum 3 skills) with validation
- ✔️ Matches preferences using TF-IDF + Cosine Similarity
- ✔️ Displays top 3 recommended career paths with match scores
- ✔️ Shows full ranking of all 10 job roles
- ✔️ Provides skill gap analysis for the top match
- ✔️ Visualises results in a bar chart

---

## 🧠 Key Skills Demonstrated
| Skill | How It Was Applied |
|---|---|
| Logic Building | Input validation, ranking logic, skill gap analysis |
| Pattern Matching | TF-IDF converts skills to weighted numerical vectors |
| Recommendation Concepts | Cosine Similarity measures alignment between user and job profiles |
| Data Handling | pandas DataFrame stores and sorts job role data |
| Visualisation | matplotlib horizontal bar chart with match scores |

---

## 🔬 How It Works — The Pipeline

```
INPUT              PROCESS                    OUTPUT
────────────────────────────────────────────────────────
User Skills   →   TF-IDF Vectorization   →   Match Scores
              →   Cosine Similarity      →   Top 3 Roles
              →   Skill Gap Analysis     →   Missing Skills
              →   Sort & Filter          →   Bar Chart
```

### Step by step:
1. User types their skills — e.g. `python, sql, machine learning`
2. TF-IDF converts both user skills and job role skills into weighted number arrays
3. Cosine Similarity measures how closely each job's skill set aligns with the user's
4. Results are sorted highest to lowest and the top 3 are displayed
5. Skill gap analysis shows what the user already has vs what they still need

---

## 🤖 Core Concepts Explained

### TF-IDF (Term Frequency — Inverse Document Frequency)
Converts text skills into numbers. Rare, specific skills (like `tensorflow`) carry **more weight** than common words (like `software`). This makes matching smarter and more precise.

### Cosine Similarity
Measures the angle between two skill vectors. A score of **1.0 = perfect match**, **0.0 = no overlap**. It is the industry standard for text-based recommendation systems.

```
Score 1.0  →  Skills perfectly aligned
Score 0.5  →  Partial match
Score 0.0  →  No skills in common
```

---

## 📊 Dataset — Job Roles Covered
| # | Job Role | Key Skills |
|---|---|---|
| 1 | Data Scientist | Python, SQL, ML, Statistics, Pandas |
| 2 | Data Analyst | SQL, Excel, Power BI, Tableau |
| 3 | ML Engineer | Python, TensorFlow, PyTorch, Deep Learning |
| 4 | Backend Developer | Python, Java, SQL, APIs, REST |
| 5 | Frontend Developer | HTML, CSS, JavaScript, React |
| 6 | DevOps Engineer | AWS, Docker, Kubernetes, Linux |
| 7 | Cloud Architect | AWS, Azure, Google Cloud, Networking |
| 8 | Cybersecurity Analyst | Security, Firewall, Linux, Encryption |
| 9 | AI Engineer | Python, Deep Learning, Neural Networks |
| 10 | Full Stack Developer | HTML, CSS, JavaScript, Python, NodeJS |

---

## 🖥️ Sample Output
```
==================================================
    AI Tech Stack Recommender
    DecodeLabs Internship | Batch 2026
==================================================
Your Skills: python, machine learning, deep learning, neural networks

✅ Skills received: python, machine learning, deep learning, neural networks

==================================================
   🏆 Top 3 Recommended Career Paths For You
==================================================

🥇 AI Engineer
   Match: 87.3%  █████████████████

🥈 Machine Learning Engineer
   Match: 74.1%  ██████████████

🥉 Data Scientist
   Match: 45.2%  █████████

==================================================
   📋 Skill Gap Analysis (Top Match)
==================================================
Role: AI Engineer
✅ Skills you have:    python, machine learning, deep learning, neural networks
📌 Skills to develop: automation, ai
```

---

## 📈 Visualisation
<img width="1485" height="879" alt="career_recommendations" src="https://github.com/user-attachments/assets/4893786a-aa9a-4ff4-b4b5-59d6f197738a" />

---

## 🛠️ Tools & Libraries
| Tool | Purpose |
|---|---|
| Python 3.x | Programming language |
| pandas | Dataset storage and sorting |
| scikit-learn | TF-IDF Vectorizer, Cosine Similarity |
| matplotlib | Bar chart visualisation |

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/DecodeLabs-AI-Projects.git
cd DecodeLabs-AI-Projects/Project3
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Recommender**
```bash
python Recommender.py
```

**4. Enter your skills when prompted**
```
Your Skills: python, sql, data analysis, machine learning, pandas
```

---

## 📁 File Structure
```
Project3/
├── Recommender1.py               # Main recommendation script
├── career_recommendations.png   # Auto-generated bar chart
└── README.md                    # This file
```

---

## 💡 Key Learnings
- **Why TF-IDF over simple matching:** Simple keyword matching treats all words equally. TF-IDF gives more weight to rare, specific skills — making recommendations smarter and more relevant.
- **Why Cosine Similarity:** It measures the direction of interest, not the size of a user's skill list. A user with 3 highly specific skills can still outmatch someone with 10 generic ones.
- **Real-world application:** This exact pipeline (TF-IDF + Cosine Similarity) powers content recommendation on major platforms before more complex neural models are applied.

---

## Intern
Odinaka R. Nwankwo 
