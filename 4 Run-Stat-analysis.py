import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, mannwhitneyu
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# الجزء الأول: اختبار فريدمان ونيميني
# ---------------------------------------------------------
def run_advanced_statistics(csv_file):
    print("\n" + "="*70)
    print(" 📊 Advanced Statistical Analysis (Friedman & Nemenyi) 📊")
    print("="*70)

    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found. Make sure training has completed first.")
        return

    df = pd.read_csv(csv_file)
    df_mean = df.groupby(['Function', 'Algorithm'])['Best'].mean().reset_index()
    pivot_df = df_mean.pivot(index='Function', columns='Algorithm', values='Best')

    data_matrix = [pivot_df[col].values for col in pivot_df.columns]
    stat, p_value = friedmanchisquare(*data_matrix)

    print(f"\n[1] Friedman Test Results (Omnibus Test):")
    print(f"    - Friedman Statistic: {stat:.4f}")
    print(f"    - P-value: {p_value:.5e}")

    if p_value < 0.05:
        print("    ✅ Statistically significant differences found. Proceeding to Nemenyi test...")
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
        nemenyi_results.columns = pivot_df.columns
        nemenyi_results.index = pivot_df.columns

        plt.figure(figsize=(12, 10))
        sns.heatmap(nemenyi_results, annot=True, cmap='coolwarm', vmin=0, vmax=0.05, fmt=".3f", linewidths=0.5)
        plt.title('Nemenyi Post-hoc Test (P-values)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        save_dir = os.path.dirname(csv_file)
        heatmap_path = os.path.join(save_dir, 'nemenyi_heatmap.png')
        plt.savefig(heatmap_path, dpi=300)
        print(f"    ✅ Nemenyi heatmap saved at: {heatmap_path}")
    else:
        print("    ❌ No statistically significant differences found. Skipping Nemenyi.")

# ---------------------------------------------------------
# الجزء الثاني: إحصاءات الأداء (W-T-L و Cohen's d)
# ---------------------------------------------------------
def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_se == 0: return 0.0
    return (np.mean(group2) - np.mean(group1)) / pooled_se

def run_performance_metrics(csv_file, target_algo='VASI_Full', alpha=0.05):
    print("\n" + "="*70)
    print(f" 📊 Advanced Performance Metrics (W-T-L, Average Rank, Cohen's d) 📊")
    print("="*70)

    df = pd.read_csv(csv_file)
    functions = df['Function'].unique()
    algorithms = df['Algorithm'].unique()

    if target_algo not in algorithms:
        print(f"❌ Target algorithm {target_algo} not found in data.")
        return

    # 1. Average Rank
    mean_best_df = df.groupby(['Function', 'Algorithm'])['Best'].mean().unstack()
    ranks_df = mean_best_df.rank(axis=1, ascending=True)  # التعديل هنا
    avg_ranks = ranks_df.mean().sort_values()

    print("\n[1] Final Average Rank (Lower is better):")
    print("-" * 40)
    for i, (algo, rank) in enumerate(avg_ranks.items(), 1):
        indicator = "⭐" if algo == target_algo else "  "
        print(f"    {i}. {indicator}{algo:<18}: {rank:.2f}")

    # 2. W-T-L and Cohen's d
    wtl_summary = {algo: {'Win': 0, 'Tie': 0, 'Loss': 0, 'Avg_Cohen_d': []} for algo in algorithms if algo != target_algo}
    effect_size_records = []

    for func in functions:
        func_data = df[df['Function'] == func]
        target_data = func_data[func_data['Algorithm'] == target_algo]['Best'].values

        for algo in algorithms:
            if algo == target_algo: continue
            competitor_data = func_data[func_data['Algorithm'] == algo]['Best'].values
            
            # منع أخطاء لو البيانات مش كاملة
            if len(target_data) == 0 or len(competitor_data) == 0: continue

            stat, p_value = mannwhitneyu(target_data, competitor_data, alternative='two-sided')
            d_value = calculate_cohens_d(target_data, competitor_data)
            wtl_summary[algo]['Avg_Cohen_d'].append(d_value)

            effect_size_records.append({'Function': func, 'Competitor': algo, 'P-value': p_value, 'Cohen_d': d_value})

            mean_target = np.mean(target_data)
            mean_competitor = np.mean(competitor_data)

            if p_value < alpha:
                if mean_target < mean_competitor: wtl_summary[algo]['Win'] += 1
                else: wtl_summary[algo]['Loss'] += 1
            else:
                wtl_summary[algo]['Tie'] += 1

    # 3. Print Summary
    print(f"\n[2] Summary of results (Win-Tie-Loss) for {target_algo} against others:")
    print("-" * 70)
    cohen_label = "Mean Cohen's d"
    print(f"    {'Competitor':<18} | {'Win (+)':<8} | {'Tie (=)':<8} | {'Loss (-)':<8} | {cohen_label:<15}")
    print("    " + "-" * 66)

    wtl_rows = []
    for algo, stats in wtl_summary.items():
        w, t, l = stats['Win'], stats['Tie'], stats['Loss']
        avg_d = np.mean(stats['Avg_Cohen_d']) if stats['Avg_Cohen_d'] else 0.0
        print(f"    {algo:<18} | {w:<8} | {t:<8} | {l:<8} | {avg_d:>.4f}")
        wtl_rows.append({'Competitor': algo, 'Win': w, 'Tie': t, 'Loss': l, 'Mean_Cohen_d': avg_d})

    # 4. Save tables
    save_dir = os.path.dirname(csv_file)
    pd.DataFrame(wtl_rows).to_csv(os.path.join(save_dir, 'wtl_summary.csv'), index=False)
    pd.DataFrame(effect_size_records).to_csv(os.path.join(save_dir, 'detailed_effect_sizes.csv'), index=False)
    print(f"\n✅ Results saved in {save_dir}/wtl_summary.csv")

if __name__ == "__main__":
    # 💡 التعديل الوحيد هنا: خلينا المسار يشاور على الفولدر بتاعك المحلي
    RESULTS_PATH = 'VASI_Results/optimization_results_runs.csv'
    
    run_advanced_statistics(RESULTS_PATH)
    run_performance_metrics(RESULTS_PATH, target_algo='VASI_Full')