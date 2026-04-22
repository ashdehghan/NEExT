.PHONY: docs clean deploy

docs:
	cd docs && make html

clean:
	cd docs && make clean

VERSION := $(shell grep -m1 '^version = ' pyproject.toml | sed -E 's/version = "([^"]+)"/\1/')
TAG     := release_v_$(subst .,_,$(VERSION))

deploy:
	@if [ ! -f .env ]; then echo "ERROR: .env not found. Create it with PYPI_API_TOKEN=pypi-..."; exit 1; fi
	@if ! grep -q '^PYPI_API_TOKEN=pypi-' .env; then echo "ERROR: PYPI_API_TOKEN in .env looks invalid (must start with pypi-)"; exit 1; fi
	@if grep -q '^PYPI_API_TOKEN=pypi-REPLACE_ME' .env; then echo "ERROR: PYPI_API_TOKEN still set to placeholder in .env"; exit 1; fi
	@if [ -n "$$(git status --porcelain)" ]; then echo "ERROR: working tree not clean. Commit or stash first."; exit 1; fi
	@BRANCH=$$(git branch --show-current); if [ "$$BRANCH" != "main" ]; then echo "ERROR: not on main (on $$BRANCH)"; exit 1; fi
	@git fetch origin main --quiet
	@if ! git merge-base --is-ancestor origin/main HEAD; then echo "ERROR: local main is behind origin/main. Pull first."; exit 1; fi
	@if git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null; then echo "ERROR: tag $(TAG) already exists locally."; exit 1; fi
	@if git ls-remote --exit-code --tags origin "refs/tags/$(TAG)" >/dev/null 2>&1; then echo "ERROR: tag $(TAG) already exists on origin."; exit 1; fi
	@echo ""
	@echo "========================================"
	@echo "  NEExT Release: $(VERSION)"
	@echo "  Tag:            $(TAG)"
	@echo "========================================"
	@echo ""
	@echo "Commits to push to origin/main:"
	@git log --oneline origin/main..HEAD || true
	@echo ""
	@echo "This will:"
	@echo "  1. git push origin main"
	@echo "  2. git tag -a $(TAG) -m 'v$(VERSION)'"
	@echo "  3. git push origin $(TAG)"
	@echo "  4. uv build  (clean dist/ first)"
	@echo "  5. uv publish  (to PyPI, using token from .env)"
	@echo ""
	@read -p "Proceed? [y/N] " ans; \
	case "$$ans" in \
	  y|Y|yes|YES) : ;; \
	  *) echo "Aborted."; exit 1;; \
	esac
	@echo ">>> Pushing main..."
	git push origin main
	@echo ">>> Tagging $(TAG)..."
	git tag -a $(TAG) -m "v$(VERSION)"
	git push origin $(TAG)
	@echo ">>> Building..."
	rm -rf dist/
	uv build
	@echo ">>> Publishing to PyPI..."
	@set -a; . ./.env; set +a; uv publish --token "$$PYPI_API_TOKEN"
	@echo ""
	@echo "DONE. Verify: https://pypi.org/project/NEExT/$(VERSION)/"
