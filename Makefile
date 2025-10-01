.PHONY: install dev test simulate clean fmt lint ci publish tag help dataset-telemachus dataset-telemachus-from

install:
	python -m pip install -r requirements.txt

dev:
	python -m pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest

simulate:
	. .venv/bin/activate && \
		RS3_ALTITUDE_CFG="/Users/sebastien.edet/projects/rs3-plugin-altitude-agpl/service/config/altitude_ign.yaml" \
		PYTHONPATH=. python -B -m runner.run_simulation2 \
		  --config config/simulator_haute_normandie.yaml

clean:
	rm -rf logs outputs .pytest_cache __pycache__

help:
	@echo "Targets:"
	@echo "  make dev        - install runtime + dev dependencies"
	@echo "  make test       - run pytest"
	@echo "  make simulate   - run example simulation (config/example.yaml)"
	@echo "  make fmt        - format code with black"
	@echo "  make lint       - lint code with flake8"
	@echo "  make ci         - local CI (fmt + lint + tests)"
	@echo "  make publish [MSG='message'] - stage+commit (if meaningful changes) + push to GitHub & Gitea"
	@echo "  make tag VERSION=x.y.z - create & push annotated git tag"

fmt:
	black .

lint:
	flake8 .

ci: fmt lint test

publish:
	@MSG="$(MSG)"; if [ -z "$$MSG" ]; then MSG="chore: auto-publish"; fi; \
	if git diff -w --quiet && git diff --cached -w --quiet; then \
	  echo "No substantial changes (ignoring whitespace). Skipping commit."; \
	else \
	  git add -A; \
	  if ! git diff --cached --quiet; then \
	    git commit -m "$$MSG"; \
	  else \
	    echo "Nothing staged to commit."; \
	  fi; \
	fi; \
	git push github HEAD:main; \
	git push gitea HEAD:main

# usage: make tag VERSION=1.0.1
tag:
	@if [ -z "$(VERSION)" ]; then echo "Usage: make tag VERSION=x.y.z"; exit 1; fi
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@git push github v$(VERSION)
	@git push gitea  v$(VERSION)

dataset-telemachus:
	@echo "🚀 Simulation RS3 (base) via runner.run_simulation2 (export Telemachus inline)"
	. .venv/bin/activate && \
		RS3_ALTITUDE_CFG="/Users/sebastien.edet/projects/rs3-plugin-altitude-agpl/service/config/altitude_ign.yaml" \
		PYTHONPATH=. python -B -m runner.run_simulation2 \
		  --config config/simulator_telemachus.yaml
	@SIM_DIR=$$(ls -dt data/simulations/TELEMACHUS_* data/simulations/simulated_* 2>/dev/null | head -1); \
	 if [ -z "$$SIM_DIR" ]; then echo "❌ Aucun dossier data/simulations/{TELEMACHUS_*, simulated_*} trouvé"; exit 2; fi; \
	 echo "✅ Dossier simulation détecté: $$SIM_DIR"; \
	 if [ -d "$$SIM_DIR/telemachus" ]; then \
	   echo "✅ Telemachus exporté inline dans: $$SIM_DIR/telemachus"; \
	   ls -l "$$SIM_DIR/telemachus" || true; \
	 else \
	   echo "⚠️ Pas de sous-dossier 'telemachus' trouvé dans $$SIM_DIR"; \
	   echo "Contenu du dossier pour debug:"; \
	   ls -la "$$SIM_DIR"; \
	 fi
