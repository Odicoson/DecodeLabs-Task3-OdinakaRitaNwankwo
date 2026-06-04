import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#--Dataset: job roles and their required skills ---
data = {
    'job_role': [
        'Data Scientist',
        'Data Analyst',
        'Machine Learning Engineer',
        'Backend Developer',
        'Frontend Developer',
        'DevOps Engineer',
        'Cloud Architect',
        'Cybersecurity Analyst',
        'AI Engineer',
        'Full Stack Developer'
    ],
    'skills': [
        'python sql machine learning data analysis statistics numpy pandas',
        'sql excel data analysis visualization power bi tableau statistics',
        'python machine learning deep learning tensorflow pytorch algorithms',
        'python java sql apis databases rest backend server',
        'html css javascript react ui ux design frontend',
        'aws docker kubernetes linux ci cd automation cloud',
        'aws cloud azure google cloud networking infrastructure architecture',
        'networking security firewall linux ethical hacking encryption',
        'python machine learning deep learning neural networks automation ai',
        'html css javascript python react nodejs databases fullstack'
    ]
}

df = pd.DataFrame(data)

# ---- Header ----
print("=" * 40)
print("    AI Tech Stack Recommender")
print("=" * 40)
print("Enter at least 3 skills separated by commas.")
print("Example: Python, Machine learning, Sql\n")

# ---- Input validation ----
while True:
    user_input = input("Your Skills: ").lower().strip()
    skills_list = [s.strip() for s in user_input.split(',') if s.strip()]
    if len(skills_list) >= 3:
        break
    print(f"⚠️  You entered {len(skills_list)} skill(s). Please enter at least 3.\n")

print(f"\n✅ Skills received: {', '.join(skills_list)}")

# ---- TF-IDF + Cosine Similarity ----
all_skills = df['skills'].tolist()
all_skills.append(user_input)

vectorizer  = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_skills)

user_vector       = tfidf_matrix[-1]
job_vectors       = tfidf_matrix[:-1]
similarity_scores = cosine_similarity(user_vector, job_vectors)[0]

df['match_score'] = similarity_scores
df_sorted         = df.sort_values('match_score', ascending=False)

# ---- Top 3 results ----
print("\n" + "=" * 50)
print("   🏆 Top 3 Recommended Career Paths For You")
print("=" * 50)

medals = ["🥇", "🥈", "🥉"]
top3   = df_sorted.head(3)

for i, (_, row) in enumerate(top3.iterrows()):
    score = round(row['match_score'] * 100, 1)
    bar   = "█" * int(score // 5)
    print(f"\n{medals[i]} {row['job_role']}")
    print(f"   Match: {score}%  {bar}")

# ---- Skill gap analysis ----
print("\n" + "=" * 50)
print("   📋 Skill Gap Analysis (Top Match)")
print("=" * 50)

top_role       = df_sorted.iloc[0]
role_skills    = set(top_role['skills'].split())
user_skills    = set(user_input.replace(',', ' ').split())
matched        = role_skills & user_skills
missing        = role_skills - user_skills

print(f"\nRole: {top_role['job_role']}")
print(f"✅ Skills you have:    {', '.join(matched) if matched else 'None matched directly'}")
print(f"📌 Skills to develop: {', '.join(missing)}")

# ---- Full ranking ----
print("\n" + "=" * 50)
print("   📊 Full Career Match Ranking")
print("=" * 50)
for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
    score = round(row['match_score'] * 100, 1)
    print(f"  {i:2}. {row['job_role']:30} {score}%")

# ---- Bar chart ----
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2ecc71' if i < 3 else '#3498db'
          for i in range(len(df_sorted))]

bars = ax.barh(
    df_sorted['job_role'],
    df_sorted['match_score'] * 100,
    color=colors,
    edgecolor='white',
    linewidth=0.5
)

# Add percentage labels on bars
for bar, (_, row) in zip(bars, df_sorted.iterrows()):
    score = round(row['match_score'] * 100, 1)
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{score}%', va='center', fontsize=9, color='white')

ax.set_xlabel('Match Score (%)', fontsize=11)
ax.set_title(f'Career Path Recommendations\nSkills: {", ".join(skills_list[:3])}{"..." if len(skills_list) > 3 else ""}',
             fontsize=12, pad=15)
ax.set_xlim(0, 110)
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend = [Patch(color='#2ecc71', label='Top 3 matches'),
          Patch(color='#3498db', label='Other roles')]
ax.legend(handles=legend, loc='lower right')

plt.tight_layout()
plt.savefig('career_recommendations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved as career_recommendations.png")
print("\nGood luck on your career journey! 🚀")
print("=" * 50)
