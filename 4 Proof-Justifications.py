import numpy as np
from scipy.special import gamma
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
from joblib import Parallel, delayed
import opfunu
import mealpy

import warnings
warnings.filterwarnings("ignore")

# استخدم مساراً محلياً داخل نفس الفولدر الذي به الكود
SAVE_DIR = "VASI_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# Test Functions (based on scientific benchmarks)
def sphere(x):
    return np.sum(x**2, axis=1)

def rastrigin(x):
    A = 10
    return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(c * x), axis=1)
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)

# ---------------------------
# Standard PSO (Kennedy & Eberhart)
# ---------------------------
def pso(N, T, d, lb, ub, obj_func, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    x = np.random.uniform(lb, ub, (N, d))
    v = np.zeros((N, d))
    pbest = x.copy()
    pbest_fitness = obj_func(x)
    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]
    fitness_history = [gbest_fitness]

    for t in range(T):
        w = w_max - (w_max - w_min) * (t / T)
        r1 = np.random.uniform(0, 1, (N, d))
        r2 = np.random.uniform(0, 1, (N, d))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = x + v
        x = np.clip(x, lb, ub)
        fitness = obj_func(x)
        update_mask = fitness < pbest_fitness
        pbest[update_mask] = x[update_mask]
        pbest_fitness[update_mask] = fitness[update_mask]
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest = pbest[current_best_idx].copy()
            gbest_fitness = pbest_fitness[current_best_idx]
        fitness_history.append(gbest_fitness)

    return gbest, gbest_fitness, fitness_history

# ---------------------------
# WSO Implementation
# ---------------------------
def wso(N, T, d, lb, ub, obj_func, a=2.0, b=0.5):
    x = np.random.uniform(lb, ub, (N, d))
    fitness = obj_func(x)
    gbest_idx = np.argmin(fitness)
    gbest = x[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]
    fitness_history = [gbest_fitness]

    for t in range(T):
        a_current = a * (1 - t / T)
        for i in range(N):
            r = np.random.uniform(0, 1)
            A = 2 * a_current * r - a_current
            C = 2 * r
            p = np.random.uniform(0, 1)
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * gbest - x[i])
                    x[i] = gbest - A * D
                else:
                    rand_whale = x[np.random.randint(0, N)]
                    D = np.abs(C * rand_whale - x[i])
                    x[i] = rand_whale - A * D
            else:
                distance_to_best = np.abs(gbest - x[i])
                # ensure numerical stability: cap distance
                distance_to_best = np.clip(distance_to_best, 1e-12, np.inf)
                x[i] = distance_to_best * np.exp(b * distance_to_best) * np.cos(2 * np.pi * distance_to_best) + gbest
            x[i] = np.clip(x[i], lb, ub)
        fitness = obj_func(x)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_fitness:
            gbest = x[current_best_idx].copy()
            gbest_fitness = fitness[current_best_idx]
        fitness_history.append(gbest_fitness)

    return gbest, gbest_fitness, fitness_history

# ---------------------------
# VASI++ Improved Implementation
# ---------------------------
def vasi_plus_plus_v2(N, T, d, lb, ub, obj_func,
                      w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
                      alpha=0.1, beta=0.2, v_th=0.5, mu=1.5, b=1.0,  # الأرقام المعتمدة
                      p_min=0.25, p_max=0.75, rho=0.8, s_levy_base=1e-3,
                      v_max_frac=0.2, use_sigmoid_p=True, k_sig=10.0, b_sig=0.5,
                      gamma_w=1.0, elitism_top_k=3, local_search_sigma=1e-3,
                      levy_trigger_patience=15, save_diag=False, diag_prefix="vasi",
                      enable_levy=True, enable_ensemble=True): # <--- الإضافة هنا
    """
    VASI++ v2: adaptive switching + dynamic Levy + hybrid gbest + local search
    - levy_trigger_patience: number of iterations with small gbest improvement to increase levy probability
    - elitism_top_k: number of top pbest used to form ensemble gbest
    """
    # init
    x = np.random.uniform(lb, ub, (N, d))
    v = np.zeros((N, d))
    pbest = x.copy()
    pbest_fitness = obj_func(x)
    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]
    fitness_history = [gbest_fitness]

    span = np.abs(ub - lb)
    # if scalar or array handle
    if np.isscalar(span):
        v_max = v_max_frac * span
    else:
        v_max = v_max_frac * np.max(span)

    # diagnostic
    diag_rows = []
    # for adaptive p(t): track recent improvements
    recent_improvements = []

    for t in range(T):
        # --- compute diversity D(t) (normalized by dimension)
        D = np.mean([np.linalg.norm(x[i] - gbest) / (np.linalg.norm(gbest) + 1e-12) for i in range(N)])

        # --- inertia schedule (with gamma)
        w = w_max - (w_max - w_min) * ((t / T) ** gamma_w) * (1 - rho * D)

        # --- form ensemble gbest_e (hybrid global best)
        # use the average of top-k pbest to reduce noisy jumps
        # 2. تعديل الـ Ensemble Global Best ليتفاعل مع الـ Flag
        if enable_ensemble:
            top_k_idx = np.argsort(pbest_fitness)[:max(1, min(elitism_top_k, N))]
            ensemble_gbest = np.mean(pbest[top_k_idx], axis=0)
            gbest_e = 0.6 * gbest + 0.4 * ensemble_gbest
        else:
            gbest_e = gbest.copy() # استخدم gbest القياسي بدون دمج

        # --- compute p_raw based on diversity and recent improvement rate
        # improvement_rate: relative improvement of gbest over last iteration
        if len(fitness_history) >= 2:
            prev = fitness_history[-1]
            improvement = (prev - gbest_fitness) / (abs(prev) + 1e-12)
        else:
            improvement = 0.0
        recent_improvements.append(improvement)
        # keep rolling window
        if len(recent_improvements) > 20:
            recent_improvements.pop(0)
        avg_improv = np.mean(recent_improvements) if recent_improvements else 0.0

        # raw p: prefer exploitation when improvement high and diversity low
        p_raw = p_min + (p_max - p_min) * (0.5 * (1 - D) + 0.5 * (1 + np.tanh(5 * avg_improv)) / 2)
        # apply sigmoid stretch to avoid extremes
        if use_sigmoid_p:
            z = k_sig * (p_raw - b_sig)
            p_t = p_min + (p_max - p_min) * (1 / (1 + np.exp(-z)))
        else:
            p_t = np.clip(p_raw, p_min, p_max)

        # dynamic levy scale: decrease as improvement rate grows; increase if stuck
        # base scale small; scale factor depends on negative avg_improv (when stuck avg_improv ~ 0 or negative)
        stuck_factor = np.clip(-avg_improv, 0.0, 1.0)  # 0 if improving, up to 1 if stuck or worsening
        s_levy = s_levy_base * (1.0 + 5.0 * stuck_factor)  # up to ~6x when stuck

        # levy trigger probability increases when no improvement for patience window
        no_improv_count = sum(1 for v_ in recent_improvements if v_ <= 1e-12)
        levy_prob_bonus = min(0.5, no_improv_count / float(max(1, levy_trigger_patience)))
        # sigma_f for decision
        sigma_f = np.std(pbest_fitness)

        levy_count = 0
        # iterate particles
        for i in range(N):
            u_switch = np.random.uniform(0, 1)
            v_candidate = np.zeros(d)

            if u_switch < p_t:
                # PSO exploitation (use ensemble gbest_e instead of raw gbest)
                r1, r2, r3 = np.random.uniform(0, 1, 3)
                v_pso = w * v[i] + c1 * r1 * (pbest[i] - x[i]) + c2 * r2 * (gbest_e - x[i])
                v_candidate = v_pso + alpha * r3 * (gbest_e - x[i])
            else:
                # WSO exploration (use ensemble too)
                a_t = 2 * (1 - t / T)
                r = np.random.uniform(0, 1)
                A = 2 * a_t * r - a_t
                C = 2 * r
                if abs(A) < 1:
                    D_vec = np.abs(C * gbest_e - x[i])
                    x_encircle = gbest_e - A * D_vec
                    v_candidate = x_encircle - x[i]
                else:
                    k_idx = np.random.randint(0, N)
                    rand_whale = x[k_idx]
                    D_vec = np.abs(C * rand_whale - x[i])
                    x_search = rand_whale - A * D_vec
                    v_candidate = x_search - x[i]
                # spiral occasionally
                if np.random.uniform(0, 1) < 0.25:
                    l = np.random.uniform(-1, 1)
                    Dg = np.abs(gbest_e - x[i])
                    Dg = np.clip(Dg, 1e-12, np.inf)
                    x_spiral = Dg * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest_e
                    v_candidate = 0.5 * (v_candidate + (x_spiral - x[i]))

            # clamp candidate velocity to avoid huge jumps
            norm_v = np.linalg.norm(v_candidate)
            if norm_v > 0:
                if norm_v > v_max:
                    v_candidate = v_candidate * (v_max / norm_v)

            # dynamic decision for Levy: use both candidate smallness and levy_prob_bonus
            # 3. تعديل قرار Levy Flight ليتوقف إذا كان الـ Flag بـ False
            levy_trigger = False
            if enable_levy: # <--- التحقق من الشرط أولاً
                if (np.linalg.norm(v_candidate) < 0.08 * v_max and sigma_f < 1e-4) or (np.random.uniform(0,1) < levy_prob_bonus * 0.2):
                    levy_trigger = True

            if levy_trigger:
                # Lévy flight (smaller scale)
                sigma_u = (gamma(1 + mu) * np.sin(np.pi * mu / 2) /
                          (gamma((1 + mu) / 2) * mu * 2**((mu - 1) / 2))) ** (1.0/mu)
                u = np.random.normal(0, sigma_u, d)
                v_norm = np.random.normal(0, 1, d)
                levy_step = u / (np.abs(v_norm) ** (1.0/mu) + 1e-12)
                x_new = x[i] + s_levy * levy_step * (gbest_e - x[i])
                levy_count += 1
            else:
                r5 = np.random.uniform(0, 1)
                x_new = x[i] + v_candidate + beta * r5 * (gbest_e - x[i])

            # boundary and evaluate
            x_new = np.clip(x_new, lb, ub)
            fitness_new = obj_func(x_new.reshape(1, -1))[0]

            # update v and x and pbest
            v[i] = v_candidate
            x[i] = x_new
            if fitness_new < pbest_fitness[i]:
                pbest[i] = x_new.copy()
                pbest_fitness[i] = fitness_new

        # local search: if no improvement in last few iters, apply small Gaussian mutation to top-elite
        if no_improv_count >= levy_trigger_patience:
            # apply to top 1 or top 2
            top_local = np.argsort(pbest_fitness)[:min(2, N)]
            for idx in top_local:
                mutation = np.random.normal(0, local_search_sigma, d) * (ub - lb)
                candidate = np.clip(pbest[idx] + mutation, lb, ub)
                f_cand = obj_func(candidate.reshape(1, -1))[0]
                if f_cand < pbest_fitness[idx]:
                    pbest[idx] = candidate.copy()
                    pbest_fitness[idx] = f_cand

        # update gbest and history
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest = pbest[current_best_idx].copy()
            gbest_fitness = pbest_fitness[current_best_idx]

        fitness_history.append(gbest_fitness)

        # diagnostics
        diag_rows.append({
            'iter': t, 'D': D, 'w': w, 'p_t': p_t, 'avg_improv': avg_improv,
            'sigma_f': sigma_f, 'gbest_fitness': gbest_fitness, 'levy_count': levy_count,
            'no_improv_count': no_improv_count
        })

    if save_diag:
        pd.DataFrame(diag_rows).to_csv(f'{diag_prefix}_diagnostics.csv', index=False)

    return gbest, gbest_fitness, fitness_history



# ---------------------------
# الدالة المُعدلة للعمل كـ Worker (عملية مستقلة 100%)
# ---------------------------
def execute_single_run(r, func_name, algo_name, algo_func_name, algo_type, N, T, d, lb, ub):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import random
    import time
    import opfunu
    import logging
    
    seed = 1000 + r
    random.seed(seed)
    np.random.seed(seed)

    funcs = opfunu.get_functions_by_classname(func_name)
    cec_base_func = funcs[0](ndim=d)
    
    def obj_func(x):
        if x.ndim == 1: return cec_base_func.evaluate(x)
        return np.array([cec_base_func.evaluate(row) for row in x])

    start_time = time.time()

    if algo_type == 'custom':
        custom_funcs = {'vasi_plus_plus_v2': vasi_plus_plus_v2, 'pso': pso, 'wso': wso}
        func_to_call = custom_funcs[algo_func_name]

        if 'VASI' in algo_name:
            e_levy = False if algo_name == 'VASI_No_Levy' else True
            e_ensemble = False if algo_name == 'VASI_No_Ensemble' else True
            _, best_fitness, history = func_to_call(N, T, d, lb, ub, obj_func, save_diag=False, enable_levy=e_levy, enable_ensemble=e_ensemble)
        else:
            _, best_fitness, history = func_to_call(N, T, d, lb, ub, obj_func)

    elif algo_type == 'mealpy':
        logging.getLogger('mealpy').setLevel(logging.WARNING)
        from mealpy import FloatVar
        import pkgutil
        import inspect
        import mealpy.evolutionary_based as eb
        import mealpy.swarm_based as sb
        
        problem_dict = {"obj_func": obj_func, "bounds": FloatVar(lb=[lb]*d, ub=[ub]*d), "minmax": "min"}
        
        # 💡 البحث العميق والشامل: فحص الملفات المادية لاستخراج الكلاس الصحيح
        targets = []
        if algo_name == 'SHADE': targets = ['OriginalSHADE', 'SHADE']
        elif algo_name == 'L-SHADE': targets = ['OriginalL_SHADE', 'OriginalLSHADE', 'L_SHADE', 'LSHADE']
        elif algo_name == 'DE': targets = ['OriginalDE', 'DE']
        elif algo_name == 'HHO': targets = ['OriginalHHO', 'HHO']
        elif algo_name == 'GWO': targets = ['OriginalGWO', 'GWO']
        elif algo_name == 'WOA': targets = ['OriginalWOA', 'WOA']
        
        opt_class = None
        for pkg in [eb, sb]:
            for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
                try:
                    mod = __import__(f"{pkg.__name__}.{modname}", fromlist=[''])
                    for name, obj in inspect.getmembers(mod, inspect.isclass):
                        if name in targets:
                            opt_class = obj
                            break
                except Exception: continue
                if opt_class: break
            if opt_class: break
            
        if not opt_class:
            raise ValueError(f"❌ Algorithm {algo_name} not found in mealpy files at all!")

        model = opt_class(epoch=T, pop_size=N)
        g_best = model.solve(problem_dict)
        best_fitness = g_best.target.fitness
        history = model.history.list_global_best_fit

    exec_time = time.time() - start_time
    return {
        'Function': func_name, 'Algorithm': algo_name, 'Run': r + 1,
        'Best': best_fitness, 'Time': exec_time, 'History': history 
    }

# ---------------------------
# دالة التجميع والتوازي الرئيسية
# ---------------------------
def compare_algorithms(runs=51, N=30, T=10000, d=30):
    import os
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from joblib import Parallel, delayed

    lb = -100.0
    ub = 100.0

    SAVE_DIR = "VASI_Results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    runs_file = os.path.join(SAVE_DIR, 'optimization_results_runs.csv')
    summary_file = os.path.join(SAVE_DIR, 'optimization_results_summary.csv')

    cec_functions = [
        "F12017", "F32017",
        "F42017", "F52017", "F82017", "F102017",
        "F112017", "F142017", "F162017", "F202017",
        "F212017", "F232017", "F262017", "F292017"
    ]

    # 💡 تم تبسيط القائمة لأسماء الخوارزميات فقط، والدالة فوق ستتكفل بإيجاد الكلاس المناسب
    algorithms_list = [
        ('L-SHADE', '', 'mealpy'),
        ('SHADE', '', 'mealpy'),
        ('HHO', '', 'mealpy'),
        ('GWO', '', 'mealpy'),
        ('DE', '', 'mealpy'),
        ('WOA', '', 'mealpy'),
        ('VASI_Full', 'vasi_plus_plus_v2', 'custom'),
        ('VASI_No_Levy', 'vasi_plus_plus_v2', 'custom'),
        ('VASI_No_Ensemble', 'vasi_plus_plus_v2', 'custom'),
        ('PSO', 'pso', 'custom'),
        ('WSO', 'wso', 'custom')
    ]

    print(f"🚀 Starting training: {len(cec_functions)} functions × {len(algorithms_list)} algorithms × {runs} runs")
    print(f"📁 Results will be automatically saved to: {SAVE_DIR}")

    for func_name in cec_functions:
        print(f"\n{'='*40}\nTesting on {func_name} function...\n{'='*40}")
        function_convergence_data = {}

        for algo_name, algo_func_name, algo_type in algorithms_list:
            print(f"  [{time.strftime('%H:%M:%S')}] Running {algo_name} (Parallel Execution)...")

            parallel_results = Parallel(n_jobs=-1, backend='loky')(
                delayed(execute_single_run)(
                    r, func_name, algo_name, algo_func_name, algo_type, N, T, d, lb, ub
                ) for r in range(runs)
            )

            best_fitnesses = [res['Best'] for res in parallel_results]
            execution_times = [res['Time'] for res in parallel_results]
            histories = [res['History'] for res in parallel_results]

            mean_time = np.mean(execution_times)
            mean_best = np.mean(best_fitnesses)
            std_best = np.std(best_fitnesses)
            min_best = np.min(best_fitnesses)
            max_best = np.max(best_fitnesses)

            summary_row = pd.DataFrame([{
                'Function': func_name, 'Algorithm': algo_name,
                'Min': min_best, 'Max': max_best,
                'Mean': mean_best, 'Std': std_best, 'Mean_Time': mean_time
            }])

            runs_data = []
            for res in parallel_results:
                runs_data.append({
                    'Function': res['Function'], 'Algorithm': res['Algorithm'],
                    'Run': res['Run'], 'Best': res['Best'], 'Time': res['Time']
                })
            df_runs = pd.DataFrame(runs_data)

            summary_row.to_csv(summary_file, mode='a', header=not os.path.exists(summary_file), index=False)
            df_runs.to_csv(runs_file, mode='a', header=not os.path.exists(runs_file), index=False)

            max_len = max(len(h) for h in histories)
            padded = [np.pad(h, (0, max_len - len(h)), 'edge') for h in histories]
            mean_history = np.mean(padded, axis=0)
            function_convergence_data[algo_name] = mean_history

        plt.figure(figsize=(10, 6))
        for algo_name, history in function_convergence_data.items():
            plt.plot(history, label=algo_name, linewidth=2)

        plt.title(f'Combined Convergence Plot - {func_name} Function')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness (log scale)')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        img_path = os.path.join(SAVE_DIR, f'convergence_{func_name}.png')
        plt.savefig(img_path, dpi=200)
        plt.close()

        print(f"  ✅ Results and plots for {func_name} function saved successfully.")

    print("\n" + "="*50)
    print("🏆 Training completed! All results safely stored.")
    print("="*50)

# Run if main
if __name__ == "__main__":
    compare_algorithms(runs=51, N=30, T=10000, d=30)
