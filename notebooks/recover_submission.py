# ===== RECOVER THE p2b SUBMISSION ====================================================
# Paste as a NEW cell in the session that finished p2b, and run it.
# No "Save Version" needed -- that is the step throwing ConcurrencyViolation.
# Gives you BOTH a download link and a push to GitHub.
import glob, gzip, os, shutil, subprocess

REPO_NAME  = globals().get("REPO_NAME",  "Business-Entity-Resolution-ML-Model")
REPO_OWNER = globals().get("REPO_OWNER", "MurtazaPatel")
REPO_DIR   = globals().get("REPO_DIR",   f"/kaggle/working/{REPO_NAME}")
PUSH_BRANCH = "submission/p2b-recovered"   # a side branch: cannot clash with main
PUSH_TO_GITHUB = True

# --- 1. locate the file --------------------------------------------------------------
tsv = f"{REPO_DIR}/output/matching_results.tsv"
if not os.path.exists(tsv):
    hits = [p for p in glob.glob("/kaggle/**/matching_results.tsv", recursive=True)
            if "/input/" not in p]
    if not hits:
        raise SystemExit("matching_results.tsv not found -- the session was recycled, so "
                         "the run has to be repeated. Nothing can recover it from here.")
    tsv = max(hits, key=os.path.getmtime)
print(f"found  {tsv}\nsize   {os.path.getsize(tsv)/1024**2:.1f} MB")

# --- 2. verify it before relying on it ----------------------------------------------
rows = non_empty = pairs = 0
with open(tsv) as f:
    header = f.readline().rstrip("\n")
    for line in f:
        rows += 1
        ids = line.rstrip("\n").partition("\t")[2]
        if ids:
            non_empty += 1
            pairs += ids.count(",") + 1
print(f"header {header!r}\nrows   {rows:,} | with predictions {non_empty:,} "
      f"({100*non_empty/max(rows,1):.1f}%) | pairs {pairs:,}")
assert header == "source1_entity_id\tmatched_entity_ids", "unexpected header"
if non_empty == 0:
    raise SystemExit("This file has NO predictions -- do not submit it.")

# --- 3. compress, and put both copies at the top of /kaggle/working ------------------
gz = "/kaggle/working/matching_results.tsv.gz"
with open(tsv, "rb") as i, gzip.open(gz, "wb", compresslevel=6) as o:
    shutil.copyfileobj(i, o, length=1 << 20)
print(f"\ngzipped -> {gz}  ({os.path.getsize(gz)/1024**2:.1f} MB)")
if os.path.abspath(tsv) != "/kaggle/working/matching_results.tsv":
    shutil.copy2(tsv, "/kaggle/working/matching_results.tsv")

# --- 4. download links: right-click -> Save link as ---------------------------------
try:
    from IPython.display import FileLink, display
    print("\nDownload directly (the .gz is ~2x faster):")
    display(FileLink("matching_results.tsv.gz"))
    display(FileLink("matching_results.tsv"))
    print("If a link 404s, use the Output panel on the right -- both files are now at "
          "the top level of /kaggle/working.")
except Exception as e:
    print("FileLink unavailable:", e)

# --- 5. push to a side branch -------------------------------------------------------
# A branch, not main: this session's clone is behind, so pushing to main would either be
# rejected as non-fast-forward or need a reset -- not worth risking on the only copy of a
# 6-hour run.
if PUSH_TO_GITHUB:
    tok = globals().get("GITHUB_TOKEN")
    if not tok:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("GITHUB_TOKEN")

    def sh(*a, check=True):
        r = subprocess.run(a, cwd=REPO_DIR, capture_output=True, text=True)
        out = (r.stdout + r.stderr).replace(tok, "***").strip()
        if out:
            print(out[:1200])
        if check and r.returncode:
            raise RuntimeError(f"git {a[1]} failed")
        return r.returncode

    shutil.copy2(gz, f"{REPO_DIR}/output/matching_results.tsv.gz")
    sh("git", "remote", "set-url", "origin",
       f"https://x-access-token:{tok}@github.com/{REPO_OWNER}/{REPO_NAME}.git")
    sh("git", "config", "user.name",  globals().get("GIT_NAME",  "Murtaza"))
    sh("git", "config", "user.email", globals().get("GIT_EMAIL", "murtazapatel05@gmail.com"))
    sh("git", "add", "-f", "output/matching_results.tsv.gz")   # -f: .gz is ignored on this commit
    sh("git", "commit", "-m", "Kaggle: p2b_baseline submission (recovered)", check=False)
    sh("git", "push", "-f", "origin", f"HEAD:refs/heads/{PUSH_BRANCH}")
    print(f"\nPUSHED to branch {PUSH_BRANCH}. On your laptop:\n"
          f"  git fetch origin\n"
          f"  git checkout origin/{PUSH_BRANCH} -- output/matching_results.tsv.gz\n"
          f"  gunzip -kf output/matching_results.tsv.gz")
