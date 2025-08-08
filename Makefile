# ================================
#  RoadSimulator3 – Makefile
# ================================

PYTHON := python3
SIM_SCRIPT := runner/run_simulation.py
CHECK_SCRIPT := core/check_realism.py
OUTPUT_DIR := data/simulations

export PYTHONPATH := .

.PHONY: simulate check clean zip release

# Lancer une simulation complète
simulate:
	@echo "🚀 Lancement de la simulation..."
	PYTHONPATH=. $(PYTHON) -m runner.run_simulation

# Vérifier la cohérence et le réalisme d'une trace
check:
	@echo "🔍 Vérification de la simulation..."
	@if [ -f $(OUTPUT_DIR)/last_trace.csv ]; then \
		PYTHONPATH=. $(PYTHON) -m core.check_realism $(OUTPUT_DIR)/last_trace.csv; \
	else \
		echo "❌ Aucune trace trouvée dans $(OUTPUT_DIR)"; \
	fi

# Nettoyer outputs, logs, cache
clean:
	@echo "🧹 Nettoyage des outputs, logs et cache..."
	bash scripts/clean_outputs.sh

# Créer une archive ZIP propre du projet
zip: clean
	@echo "📦 Création de l'archive RoadSimulator3.zip..."
	@zip -r RoadSimulator3.zip . \
		-x "*.git*" \
		-x "__pycache__/*" \
		-x "*.pyc" \
		-x "*.pyo" \
		-x "*.DS_Store" \
		-x "data/*" \
		-x "notebooks/*" \
		-x "outputs/*" \
		-x "logs/*" \
		-x "out*/**" \
		-x "cache/*" \
		-x "venv/*" \
		-x ".venv/*"

# Release complète avec commit, tag, push et archive
release:
	@if [ -z "$(VERSION)" ]; then \
		echo "❌ VERSION non spécifiée. Utilise : make release VERSION=1.0.0"; \
		exit 1; \
	fi
	@echo "📦 Publication version $(VERSION)..."
	@git add .
	@git commit -m "🚀 Version $(VERSION) stable" || echo "(info) aucun changement à committer"
	@git tag -a v$(VERSION) -m "Version stable v$(VERSION) – RoadSimulator3"
	@git push origin main
	@git push origin v$(VERSION)
	$(MAKE) zip