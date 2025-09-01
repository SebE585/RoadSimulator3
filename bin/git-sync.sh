#!/usr/bin/env bash
set -euo pipefail

# === Réglages par défaut ===
REMOTE_URL_DEFAULT="git@ulysse:seb/RoadSimulator3.git"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [-m "message"] [-b branche] [-t tag|auto] [-r remote_url] [-C path]
  -m  Message de commit (défaut: "chore: sync <date>")
  -b  Branche à pousser (défaut: branche courante)
  -t  Tag à créer et pousser (ex: v0.1.2), ou 'auto' pour vYYYYmmdd-HHMMSS
  -r  URL du remote origin (défaut: ${REMOTE_URL_DEFAULT})
  -C  Chemin du repo (défaut: répertoire courant)
Exemples:
  $(basename "$0") -m "feat: tuning inertial noise"
  $(basename "$0") -b feat/pipeline-noise-tuning -m "fix: clamp speed >= 0"
  $(basename "$0") -t auto
USAGE
}

COMMIT_MSG=""
BRANCH=""
TAG_NAME=""
REMOTE_URL=""
REPO_DIR=""

# --- Parse options ---
while getopts ":m:b:t:r:C:h" opt; do
  case $opt in
    m) COMMIT_MSG="$OPTARG" ;;
    b) BRANCH="$OPTARG" ;;
    t) TAG_NAME="$OPTARG" ;;
    r) REMOTE_URL="$OPTARG" ;;
    C) REPO_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Option invalide: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "Option -$OPTARG requiert un argument." >&2; usage; exit 2 ;;
  esac
done

# --- Aller dans le repo ---
if [[ -n "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "❌ Pas un repo git."; exit 1; }

# --- Déterminer la branche ---
current_branch="$(git rev-parse --abbrev-ref HEAD)"
branch="${BRANCH:-$current_branch}"

if [[ "$branch" != "$current_branch" ]]; then
  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    git checkout "$branch"
  else
    git checkout -b "$branch"
  fi
fi

# --- Config remote origin si absent ---
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "${REMOTE_URL:-$REMOTE_URL_DEFAULT}"
fi

# --- Add & commit si nécessaire ---
git add -A

if ! git diff --cached --quiet; then
  msg="${COMMIT_MSG:-"chore: sync $(date +'%Y-%m-%d %H:%M:%S')"}"
  git commit -m "$msg"
else
  echo "ℹ️  Aucun changement à committer."
fi

# --- Pousser la branche (avec upstream si manquant) ---
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  # upstream existe déjà
  git push
else
  git push -u origin "$branch"
fi

# --- Tag optionnel ---
if [[ -n "${TAG_NAME}" ]]; then
  if [[ "${TAG_NAME}" == "auto" ]]; then
    TAG_NAME="v$(date +'%Y%m%d-%H%M%S')"
  fi
  # Crée le tag si absent, sinon le met à jour (annoté)
  if git rev-parse "refs/tags/${TAG_NAME}" >/dev/null 2>&1; then
    echo "ℹ️  Le tag ${TAG_NAME} existe déjà localement, re-création."
    git tag -d "${TAG_NAME}" >/dev/null 2>&1 || true
  fi
  git tag -a "${TAG_NAME}" -m "${COMMIT_MSG:-"tag ${TAG_NAME}"}"
  git push origin "${TAG_NAME}"
  echo "🏷️  Tag poussé: ${TAG_NAME}"
fi

echo "✅ Push OK sur branche ${branch}"
