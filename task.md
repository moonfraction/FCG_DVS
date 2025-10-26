## **Summary of the FCG Algorithm Implementation**

### **Objective**

Mitigate fairness bias in Large Language Models (LLMs) via representative sample selection and evolution-based optimization, combining clustering (for diversity) and genetic evolution (for fairness-performance balance).

---

### **Algorithm Overview**

The process consists of two main stages:

1. **Diverse Clustering (STEP 1)**
2. **Genetic Evolution and Score Update (STEP 2)**

---

### **STEP 1: Diverse Clustering**

**Goal:**
Create a diverse and representative subset of samples (SG′) from the training dataset (`Dtr`) based on sensitive feature `Z` and label `Y`.

**Procedure:**

1. Split training data `Dtr` into **four subgroups** based on combinations of `Z` and `Y`:

   ```
   SG = { g1(Z=1,Y=0), g2(Z=1,Y=1), g3(Z=0,Y=0), g4(Z=0,Y=1) }
   ```
2. For each subgroup `gi`:

   * Apply **K-Means clustering** with `n` clusters.
   * For each cluster centroid:

     * Compute distances of all points to the centroid.
     * Select the **`m` closest samples** as representative neighbors.
3. Combine all selected samples into new subgroups:

   ```
   SG' = { g1', g2', g3', g4' }
   ```
4. These subgroups serve as a **diverse initial population** for the next evolutionary step.

---

### **STEP 2: Genetic Evolution and Score Update**

**Goal:**
Iteratively evolve the demonstration set based on **prediction performance** and **fairness metrics**.

**Procedure:**

1. Initialize **score `p`** for all samples.
2. For each subgroup `g'i` in `SG'`:

   * Run a **genetic algorithm** (`roulette wheel selection`) for `iters` iterations.
   * In each iteration:

     * Select `K` demonstration samples.
     * Compute the **Evolutionary Score (`EvolScore`)** for each using the function `CAL_SCORE()`.
     * Update sample scores with averaged `EvolScore` when selected.
3. **Higher-scoring samples** are more likely to be selected in subsequent iterations.

---

### **CAL_SCORE Function**

**Inputs:**

* Selected samples (`shots`)
* Evaluation criteria `ch = [M_pred, M_fair]`

**Process:**

1. Get base LLM predictions on dev set `X_dev` → compute baseline performance (`Base_pred`, `Base_fair`).
2. Get LLM predictions conditioned on selected samples → compute improved metrics (`ICL_pred`, `ICL_fair`).
3. Compute performance improvements:

   ```
   Δpred = max(ICL_pred - Base_pred, p)
   Δfair = max(ICL_fair - Base_fair, p)
   ```
4. Compute combined evolution score:

   ```
   EvolScore = α * Δpred + (1 - α) * Δfair
   ```

   where `α` balances prediction and fairness (default α = 0.5).

---

### **Testing Phase**

1. On test data (`Dtest`), rank demonstrations in `SG′` by their **average `EvolScore`**.
2. Select **top-performing demonstrations** per subgroup for in-context learning (ICL).
3. Evaluate final LLM fairness and performance metrics.

---

### **Evaluation Metrics**

**Predictive Metrics:**

* Accuracy, Precision, Recall, F-score

**Fairness Metrics:**

1. **Demographic Parity (DP):**

   * ( DP_0 = P(f(x)=1|Z=0) )
   * ( DP_1 = P(f(x)=1|Z=1) )
   * **Δdp** = |DP₁ − DP₀|
   * **Rdp** = DP₀ / DP₁

2. **Equalized Odds (Eodds):**

   * ( TPR_z = P(f(x)=1|y=1,Z=z) )
   * ( FPR_z = P(f(x)=1|y=0,Z=z) )
   * **Δeo** = max(|TPR₁−TPR₀|, |FPR₁−FPR₀|)
   * **Reo** = min(TPR₀/(TPR₁+ϵ), FPR₀/(FPR₁+ϵ))

**Interpretation:**

* Lower Δdp, Δeo → Higher fairness
* Higher Rdp, Reo → More consistent subgroup performance

---

### **Prompt Structure for LLM Evaluation**

1. **Task Description:** Defines prediction task and output label format.
2. **In-Context Demonstrations:** Reference examples selected using FCG.
3. **Question:** The test sample to predict.

* **Zero-shot**: Uses only parts (1) & (3).
* **Few-shot / ICL**: Uses all three parts.

---

### **Experimental Setup**

* Dataset: **Adult**
* Clusters (`n`): 8
* Neighbors per cluster (`m`): 5
* Initial score (`p`): 0.05
* Iterations: 10
* Scoring metrics: `M_pred = F-score`, `M_fair = Reo`
* Balance coefficient (`α`): 0.5

---

### **Implementation Summary**

* Input: `Dtr`, `Ddev`, `Dtest`, sensitive feature `Z`, label `Y`
* Output: Ranked list of demonstrations with updated `EvolScores`
* Core modules:

  1. Clustering (K-Means)
  2. Genetic algorithm (Roulette Wheel)
  3. Score computation (`CAL_SCORE`)
  4. Fairness evaluation (`Δdp`, `Δeo`, `Rdp`, `Reo`)

The Workflow of Fairness via Clustering-Genetic (FCG) on the Adult Dataset (ry = 1, all high-income;
ry = 0, all low-income; ry = 0.5, balanced samples of high-income and low-income; rz = 1, all females; rz = 0,
all males; rz = 0.5, balanced samples of females and males.)


here are some sample code you can modify and use
def loadAdult():
    tr_path = 'ds/adult/adult.data'
    test_path = 'ds/adult/adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    drop_names = ['education-num', 'fnlwgt', 'race', 'native-country']

    dftr = readAndProcess(tr_path, column_names, drop_names)
    dftst = readAndProcess(test_path, column_names, drop_names)
    return dftr,dftst

def convertAdultIncome(dfin):
  df=dfin.copy(deep=True)
  df.rename(columns={'income': 'income answer'}, inplace=True)
  df['income answer'] = df['income answer'].replace('<=50K', 'less than or equal to 50K')
  df['income answer'] = df['income answer'].replace('>50K', 'greater than 50K')
  return df



### **Algorithm 1: FCG Algorithm**

#### **STEP 1: Diverse Clustering**

```text
procedure STEP1: DIVERSE_CLUSTERING
    all_idx_set = []
    for each gi in SG do
        selected_idx_set = []
        X = gi.get_all_data()
        centroids = KMeans(clusters = n).fit(X)
        for each center in centroids do
            dis_set = distance(X, center)
            closest = argsort(dis_set)[:m]
            selected_idx_set.extend(closest)
        end for
        all_idx_set.append(selected_idx_set)
    end for
    return SG’.set(all_idx_set)
end procedure
```

---

#### **STEP 2: Update Evolution Score**

```text
procedure STEP2: UPDATE_EVOL_SCORE
    ch = [M_pred, M_fair]
    for each g′i in SG′ do
        shots = g′i.genetic_algorithm(k)
        evol_score = CAL_SCORE(shots, ch)
        shots.update(evol_score)
        repeat iters times
    end for
end procedure
```

---

#### **CAL_SCORE Function**

```text
function CAL_SCORE(shots, ch)
    Y_base = LLM.predict(X_dev)
    Base_pred, Base_fair = eval(Y_base, Y_dev, ch)
    
    Y_ICL = LLM.predict(shots, X_dev)
    ICL_pred, ICL_fair = eval(Y_ICL, Y_dev, ch)
    
    Δpred = max((ICL_pred - Base_pred), p)
    Δfair = max((ICL_fair - Base_fair), p)
    
    return α · Δpred + (1 − α) · Δfair
end function
```

---

use groq api using openai config
import os
import openai
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)


implement the above task as mentioned in description
