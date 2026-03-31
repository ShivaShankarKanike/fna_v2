from .fluid_dynamics import advect, diffuse, project, ns_step
from .adapter import FNALayer
from .optimizer import FNAOptimizer
from .injection import inject_fna_adapters, get_fna_optimizer_params, print_model_summary

__all__ = [
    "advect", "diffuse", "project", "ns_step",
    "FNALayer", "FNAOptimizer",
    "inject_fna_adapters", "get_fna_optimizer_params", "print_model_summary",
]
```

4. Click **Commit changes**

---

**Now upload the 4 fna module files:**

1. Click **Add file → Upload files**
2. Before dropping files, click into the `fna/` folder first (click `fna` in the file list)
3. Drag and drop these 4 files:
   - `fluid_dynamics.py`
   - `adapter.py`
   - `optimizer.py`
   - `injection.py`
4. Click **Commit changes**

---

**Now create the experiments folder:**

1. Click **Add file → Create new file**
2. In the filename box type: `experiments/__init__.py`
3. Leave the content box empty (just one space)
4. Click **Commit changes**

---

**Now upload the experiment file:**

1. Click into the `experiments/` folder
2. Click **Add file → Upload files**
3. Drop `run_mmlu.py`
4. Click **Commit changes**

---

**Verify your repo structure looks like this:**
```
fna_v2/
├── fna/
│   ├── __init__.py
│   ├── fluid_dynamics.py
│   ├── adapter.py
│   ├── optimizer.py
│   └── injection.py
└── experiments/
    ├── __init__.py
    └── run_mmlu.py
