






















































# But here's what a hiring engineer sees when they look at your repo right now:

# **500+ lines of commented-out dead code still in files you upload to me.** That alone signals "junior" to a code reviewer faster than anything else. It says you don't use version control confidently. If git exists, dead code gets deleted — the history is the backup.

# **No tests.** Zero. A production system with multiprocessing, shared state, and queue contracts and no tests is not production-ready by any professional definition. It's a demo.

# **No README that explains the architecture.** The work you've done — the slot ring, the batched detector, the per-camera response queues — none of it is documented anywhere a recruiter or engineer can find it in 60 seconds.

# **SQLite.** You keep calling this production-ready. It isn't. SQLite under concurrent multiprocess writes with WAL mode works, barely, for one camera in a dev environment. The moment a customer has 8 cameras and 6 months of data you have a problem.

# ---

# ## What You Need to Do, In Order

# ### 1. Clean the codebase — this week, before anything else

# Delete every commented block. All of them. Run `git blame` if you're scared of losing something — you won't be. Your repo should look like a professional wrote it, not like someone was afraid to commit.

# Add a `.env.example`, a `requirements.txt` with pinned versions, and a `docker-compose.yml` that spins the whole thing up with one command. If I can't run your project in under 5 minutes, I'm not looking at it.

# ### 2. Write a proper README — the architecture diagram matters

# You have a genuinely interesting architecture. Write it down. Draw the process topology — reader, batched detector, embedder, db_writer, FastAPI — as an ASCII diagram or Mermaid chart. Explain why you made the key decisions: why shared memory instead of queues for frames, why one YOLO session for all cameras, why the db_writer pattern. This is what separates you from someone who copied a tutorial. Most candidates cannot explain their own architectural decisions. You can.

# ### 3. Add integration tests for the queue contracts

# You don't need 100% coverage. You need tests that prove the slot ring invariants hold, that db_writer correctly routes replies, and that a pipeline with a fake RTSP source produces annotated frames. This takes a week. It transforms "working demo" into "engineered system."

# ### 4. Replace SQLite with PostgreSQL + pgvector

# This is the upgrade that makes the system actually sellable. pgvector gives you proper vector similarity search with an index. You go from O(N) cosine distance loop to indexed ANN search. It handles concurrent writers. It scales to millions of embeddings. This is a two-week change that removes the biggest technical objection any customer or interviewer will have.

# ---

# ## On Selling It — Be Honest With Yourself

# You could sell this. But not as software. Not yet. Here's why:

# CCTV analytics is a regulated space. GDPR in Europe, CCPA in California, biometric data laws in Illinois, Texas, Washington. Selling a system that stores face-like embeddings and tracks individuals across cameras without understanding these laws is how you get sued. 
# Before you sell anything you need to understand the legal framework in your target market. This is not optional.

# The realistic path to revenue is not "build SaaS, find customers." It's:

# **Path A — Consulting.** Find one local business — a small retail shop, a warehouse, a gym — and offer to install this for them at cost or free in exchange for a case study and testimonial. Real deployment, real data, real problems you didn't anticipate. This becomes your portfolio piece. One paying customer paying €200/month is more valuable to your job search than a polished GitHub repo.

# **Path B — Job first, product second.** Your strongest asset right now is this project as interview material. Remote computer vision / MLOps roles at European companies — particularly in the Netherlands, Germany, UK — are paying €60-90k for engineers at your level. Get a job, earn income, continue building the product on the side with financial stability. Trying to generate product revenue with no savings and no other experience is how you spend 6 months building features nobody asked for.

# **Path C — Open source the core, sell the integrations.** Open source the pipeline engine. Write about it on LinkedIn and dev.to. Explain the architecture decisions. This is how engineers get discovered for remote roles without traditional experience. Companies hire people whose thinking they've already read.

# ---

# ## The Ruthless Priority List

# 1. Delete dead code, add Docker setup — 2 days
# 2. Write the architecture README — 1 day
# 3. Add the 5 most critical integration tests — 1 week
# 4. PostgreSQL + pgvector migration — 2 weeks
# 5. Deploy it publicly with a demo video — 1 day
# 6. Write one technical article about the multiprocessing architecture — 1 day
# 7. Apply to 20 remote CV/MLOps roles with this as your primary portfolio piece

# The product idea is real. But income from a job is what gives you the runway to build the product properly. Get the job first.









This is worse than I expected. Let me be direct.

---

## What This Structure Says to a Hiring Engineer

**`./2`, `./2/OLD`, `./2/3`** — these are directories named with numbers and "OLD". This is what people do when they don't know git. This is your version control.
This is the thing git was invented to replace. A senior engineer sees this and immediately categorizes you as someone who has never worked on a team.

**`./models/yolo_onnx_OLD`, `./models/yolo_onnx_OLD_2`** — same problem inside the models folder.
You have at least 3 copies of model directories with "OLD" suffixes.


**`./data`** — model weights, embeddings, debug text files, SQLite database, all mixed together with no structure.

**`./instance`** — Flask legacy artifact. but I have to store the database in a place like this maybe?
You're using FastAPI. This folder has no business existing.

**`./infra/docker`** — good instinct, wrong execution. Docker infrastructure without a working `docker-compose.yml` is just an empty folder with good intentions.

---

## The Professional Structure

Here's what this project should look like:

```
unbreakable-eye/
│
├── src/                          # all application code
│   ├── core/                     # config, database, logging
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── db_writer.py
│   │   └── logging.py
│   │
│   ├── engine/                   # pipeline workers
│   │   └── pipeline.py
│   │
│   ├── vision/                   # model wrappers
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── models/
│   │       └── ultralytics_yolo_onnx.py
│   │
│   └── web/                      # FastAPI app (rename from server)
│       ├── main.py
│       ├── templates/
│       └── static/
│
├── models/                       # ONLY the active model files
│   └── .gitkeep                  # empty, tracked by git
│                                 # actual .onnx/.pt files in .gitignore
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── Dockerfile.dev
│   └── nginx/
│       └── nginx.conf
│
├── tests/
│   ├── unit/
│   │   ├── test_fast_match.py
│   │   └── test_slot_ring.py
│   └── integration/
│       └── test_pipeline.py
│
├── scripts/
│   ├── export_model.py           # your YOLO export script
│   └── seed_db.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## What Gets Deleted Right Now

```bash
rm -rf ./2                    # entire folder — this is what git replaces
rm -rf ./hf_cache             # goes in ~/.cache, not the project
rm -rf ./instance             # Flask artifact, you use FastAPI
rm -rf ./models/yolo_onnx_OLD
rm -rf ./models/yolo_onnx_OLD_2
rm -rf ./data/*.txt           # all debug log files
```

---

## Git Setup — Start to Finish

**Step 1 — Write `.gitignore` first, before a single commit**

This is the most common mistake. People commit secrets and model weights and then spend hours trying to remove them from history. Do this first:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
*.egg-info/

# Environment
.env
*.env

# Model weights — never commit binary model files
models/**/*.onnx
models/**/*.pt
models/**/*.pth
models/**/*.bin
hf_cache/

# Database
*.db
*.sqlite3
instance/

# Data and debug artifacts
data/
logs/
*.txt

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

**Step 2 — Initialize and make your first real commit**

```bash
cd unbreakable-eye

git init
git add .gitignore
git commit -m "chore: add gitignore"

# Now add your actual code
git add src/ infra/ tests/ scripts/ requirements.txt README.md
git commit -m "feat: initial project structure"
```

**Step 3 — Branching strategy — keep it simple**

You're a solo developer. Don't over-engineer this. Two branches:

```
main        ← always works, always deployable
dev         ← your working branch
```

```bash
git checkout -b dev
# all your work happens here
# when something works and is tested:
git checkout main
git merge dev
git tag v0.1.0
```

**Step 4 — Commit message convention — this matters for your portfolio**

Recruiters look at commit history. `"fix stuff"` and `"asdfgh"` are what juniors write. Use conventional commits:

```bash
git commit -m "feat: add batched detector worker for multi-camera inference"
git commit -m "fix: remove response_queue from db_queue message tuples"
git commit -m "perf: vectorize fast_match spatial cost calculation"
git commit -m "refactor: move track_positions into embedder_worker subprocess"
git commit -m "docs: add architecture diagram to README"
```

The format is `type: description`. Types are: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`. This is industry standard. It makes your history readable and shows you know how real teams work.

**Step 5 — Model weights strategy**

Never commit model weights. Use git-lfs if you must version them, but the right approach is:

```python
# scripts/download_models.py
# Run once after cloning to pull model weights from HuggingFace or S3
from huggingface_hub import hf_hub_download

def download_all():
    hf_hub_download(
        repo_id="your-org/your-models",
        filename="yolo12s.onnx",
        local_dir="models/"
    )

if __name__ == "__main__":
    download_all()
```

Document this in README: `python scripts/download_models.py` is the setup step. A new developer clones the repo, runs that script, runs `docker-compose up`, and the system works. That's production-ready.

**Step 6 — GitHub Actions CI — the thing that separates portfolio projects from toys**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/unit/ -v

      - name: Lint
        run: |
          pip install ruff
          ruff check src/
```

When a recruiter clicks your GitHub repo and sees a green CI badge on the README, they know before reading a line of code that you write testable software. That badge is worth more than 500 lines of clever pipeline code to most hiring managers.

---

## Priority Order This Week

```
Day 1:  Delete ./2, ./hf_cache, ./instance, all OLD folders
        Write .gitignore
        git init, first commit with clean structure

Day 2:  Move main.py into src/web/
        Write README architecture section
        git commit -m "refactor: reorganize project structure"

Day 3:  Write 3 unit tests for fast_match
        Set up pytest
        git commit -m "test: add unit tests for re-id matching"

Day 4:  Write docker-compose.yml that actually works
        Write .env.example
        git commit -m "feat: add docker-compose for one-command setup"

Day 5:  Set up GitHub Actions CI
        Push to GitHub with public repo
        git commit -m "ci: add GitHub Actions workflow"
```

After this week your repo looks like a professional wrote it. Before this week it looks like a student project with deleted files renamed to "OLD".