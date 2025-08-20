# NEExT Manuscript Makefile
# Convenient commands for building and managing the manuscript

.PHONY: all build watch clean open install help

# Default TeX file
TEX_FILE = neext_reddit.tex
PDF_FILE = neext_reddit.pdf

# Default target
all: build

# Build the manuscript
build:
	@echo "📚 Building manuscript..."
	@python3 build_manuscript.py --tex $(TEX_FILE)

# Build and open PDF
open: build
	@python3 build_manuscript.py --open --tex $(TEX_FILE)

# Watch for changes and auto-rebuild
watch:
	@echo "👁️  Starting watch mode..."
	@python3 build_manuscript.py --watch --tex $(TEX_FILE)

# Watch and open initially
watch-open:
	@python3 build_manuscript.py --watch --open --tex $(TEX_FILE)

# Clean auxiliary files
clean:
	@echo "🧹 Cleaning auxiliary files..."
	@python3 build_manuscript.py --clean
	@rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.synctex.gz
	@echo "✅ Clean complete"

# Install Tectonic (modern LaTeX engine)
install-tectonic:
	@echo "📦 Installing Tectonic..."
	@if command -v brew >/dev/null 2>&1; then \
		brew install tectonic; \
	elif command -v conda >/dev/null 2>&1; then \
		conda install -c conda-forge tectonic -y; \
	else \
		echo "Downloading Tectonic binary..."; \
		curl -L https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-macOS-latest.tar.gz | tar xz; \
	fi

# Install MacTeX (full LaTeX distribution for macOS)
install-mactex:
	@echo "📦 Installing MacTeX (this is large ~4GB and will take time)..."
	@if command -v brew >/dev/null 2>&1; then \
		brew install --cask mactex; \
	else \
		echo "Please download MacTeX from: https://tug.org/mactex/"; \
	fi

# Quick PDF viewer
view:
	@if [ -f $(PDF_FILE) ]; then \
		if command -v open >/dev/null 2>&1; then \
			open $(PDF_FILE); \
		elif command -v xdg-open >/dev/null 2>&1; then \
			xdg-open $(PDF_FILE); \
		else \
			echo "Please open $(PDF_FILE) manually"; \
		fi \
	else \
		echo "❌ PDF not found. Run 'make build' first."; \
	fi

# Docker build (most reliable)
docker-build:
	@echo "🐳 Building with Docker..."
	@docker run --rm -v "$$(pwd)":/manuscript -w /manuscript texlive/texlive:latest \
		sh -c "pdflatex -interaction=nonstopmode $(TEX_FILE) && \
		       bibtex $$(basename $(TEX_FILE) .tex) || true && \
		       pdflatex -interaction=nonstopmode $(TEX_FILE) && \
		       pdflatex -interaction=nonstopmode $(TEX_FILE)"

# Create a ZIP for Overleaf upload
overleaf-zip:
	@echo "📦 Creating ZIP for Overleaf..."
	@zip -r manuscript_for_overleaf.zip *.tex *.bib *.cls *.bst *.jpg *.png
	@echo "✅ Created manuscript_for_overleaf.zip"
	@echo "📝 Upload this to Overleaf: https://www.overleaf.com/project"

# Help
help:
	@echo "NEExT Manuscript Build System"
	@echo "============================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build         - Build the PDF"
	@echo "  make open          - Build and open the PDF"
	@echo "  make watch         - Watch for changes and auto-rebuild"
	@echo "  make watch-open    - Watch mode with initial PDF open"
	@echo "  make clean         - Remove auxiliary files"
	@echo "  make view          - Open existing PDF"
	@echo "  make docker-build  - Build using Docker (reliable)"
	@echo "  make overleaf-zip  - Create ZIP for Overleaf upload"
	@echo ""
	@echo "Installation commands:"
	@echo "  make install-tectonic - Install Tectonic (lightweight)"
	@echo "  make install-mactex   - Install MacTeX (full distribution)"
	@echo ""
	@echo "Current TeX file: $(TEX_FILE)"
	@echo "Output PDF: $(PDF_FILE)"